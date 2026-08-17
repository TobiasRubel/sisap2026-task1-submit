#pragma once

#include <math.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#include "../bench/parse_command_line.h"
#include "../utils/graph.h"
#include "../utils/h5_result_writer.h"
#include "pipnn_utils.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

namespace parlayANN {
std::atomic<size_t> leaf_count(0);

template <typename GraphH, typename Point, typename PointRange, typename indexType>
struct cluster {
  using distanceType = typename Point::distanceType;
  using edge = std::pair<indexType, indexType>;
  using labelled_edge = std::pair<edge, distanceType>;
  using GraphI = Graph<indexType>;
  using PR = PointRange;
  using Bucket = parlay::sequence<uint32_t>;

  cluster() {}

  parlay::sequence<Bucket> SelectAndAssignLeaders(
      Bucket& ids, int depth, size_t seed) {
    size_t num_leaders =
        (depth == 0) ? TOP_LEVEL_NUM_LEADERS : ids.size() * FRACTION_LEADERS;
    num_leaders = std::min<size_t>(num_leaders, MAX_NUM_LEADERS);
    num_leaders = std::max<size_t>(num_leaders, 3);

    COMPARISONS.fetch_add(num_leaders * ids.size(), std::memory_order_relaxed);

    Bucket leaders;
    {
      std::mt19937 prng(seed);
      leaders.resize(num_leaders);
      std::sample(ids.begin(), ids.end(), leaders.begin(), leaders.size(),
                  prng);
    }

    int fanout = 1;
    if (depth < (int)FANOUT_SCHEME.size()) {
      fanout = FANOUT_SCHEME[depth];
    }
    fanout = std::min<int>(fanout, (int)num_leaders);

    parlay::sequence<std::pair<uint32_t, uint32_t>> flat(ids.size() * fanout);
    parlay::internal::timer t;
    t.start();
    bool used_hamming = false;
    if (HAMMING_ASSIGN && rsqpr1_ && rsqpr_ &&
        depth >= 1 && fanout < 10) {
      rsq_assign_leaders_hamming_rescore(*rsqpr1_, *rsqpr_, ids, leaders, fanout,
                                        flat, depth, ASSIGN_MARGIN_TAU,
                                        &ASSIGN_EMITTED_);
      ASSIGN_SLOTS_.fetch_add(ids.size() * fanout, std::memory_order_relaxed);
      used_hamming = true;
    } else if (rsqpr_) {
      rsq_assign_leaders(*rsqpr_, ids, leaders, fanout, flat, depth);
    } else {
      std::cerr << "FATAL: leader assignment requires the RSQ index."
                << std::endl;
      std::abort();
    }
    if (used_hamming) {
      flat = parlay::filter(flat, [](const std::pair<uint32_t, uint32_t>& p) {
        return p.first != kInvalidLeader;
      });
    }
    if (depth == 0) {
      t.next("first level assign time");

      if (CAPTURE_CELL0 && CELL0_.empty()) {
        CELL0_ = parlay::sequence<uint32_t>(ids.size(), 0);
        parlay::parallel_for(0, ids.size(), [&](size_t i) {
          const auto& slot0 = flat[i * fanout];
          CELL0_[slot0.second] = slot0.first;
        });
      }
    }

    parlay::sequence<Bucket> clusters =
        parlay::group_by_index(flat, leaders.size());
    leaders.clear();

    clusters = parlay::filter(
        clusters, [](const Bucket& b) { return b.size() > 0; });

    return clusters;
  }

  void MergeSmallClusters(parlay::sequence<Bucket>& clusters) {
    std::sort(
        clusters.begin(), clusters.end(),
        [](const Bucket& a, const Bucket& b) { return a.size() > b.size(); });
    while (clusters.size() > 1 &&
           clusters.back().size() < MIN_CLUSTER_SIZE) {
      Bucket small_cluster = std::move(clusters.back());
      clusters.pop_back();
      int merge_target_idx = -1;
      for (int i = clusters.size() - 1; i >= 0; --i) {
        if (clusters[i].size() + small_cluster.size() <=
            MAX_CLUSTER_SIZE) {
          merge_target_idx = i;
          break;
        }
      }

      if (merge_target_idx != -1) {
        clusters[merge_target_idx].insert(clusters[merge_target_idx].end(),
                                         small_cluster.begin(),
                                         small_cluster.end());
        clusters[merge_target_idx] =
            parlay::remove_duplicates(clusters[merge_target_idx]);
        int current_idx = merge_target_idx;
        while (current_idx > 0 && clusters[current_idx].size() >
                                      clusters[current_idx - 1].size()) {
          std::swap(clusters[current_idx], clusters[current_idx - 1]);
          current_idx--;
        }
      } else {
        clusters.push_back(std::move(small_cluster));
        break;
      }
    }
  }

  parlay::sequence<Bucket> SplitClusterRandomly(Bucket& cluster) {
    std::mt19937 prng(cluster.size());
    std::shuffle(cluster.begin(), cluster.end(), prng);
    parlay::sequence<Bucket> splits(1);
    for (uint32_t u : cluster) {
      if (splits.back().size() == MAX_CLUSTER_SIZE) {
        splits.emplace_back();
      }
      splits.back().push_back(u);
    }
    return splits;
  }

  template <typename ProcessLeafFn>
  void ClusterRecursively(
      Bucket &cluster, int depth,
      const ProcessLeafFn& process_leaf) {
    if (cluster.size() <= MAX_CLUSTER_SIZE) {
      process_leaf(std::move(cluster));
      return;
    }

    parlay::sequence<Bucket> clusters = SelectAndAssignLeaders(
        cluster, depth, parlay::hash64(SEED + cluster.size() + depth));

    const size_t num_points = cluster.size();
    cluster.resize(0);

    MergeSmallClusters(clusters);

    size_t gran = 1;
    if (!clusters.empty()) {
      size_t child_sum = 0;
      for (const auto& c : clusters) child_sum += c.size();
      const size_t avg_child =
          std::max<size_t>(1, child_sum / clusters.size());
      if (avg_child <= MAX_CLUSTER_SIZE) {
        static constexpr size_t kLeafBatchPoints = 4096;
        gran = std::min<size_t>(
            16, std::max<size_t>(1, kLeafBatchPoints / avg_child));
      }
    }
    parlay::parallel_for(
        0, clusters.size(),
        [&](size_t c) {
          Bucket& my_cluster = clusters[c];
          if (depth > MAX_DEPTH ||
              (depth > CONCERNING_DEPTH &&
               my_cluster.size() >
                   TOO_SMALL_SHRINKAGE_FRACTION * num_points)) {
            auto splits = SplitClusterRandomly(my_cluster);
            for (auto& s : splits) {
              process_leaf(std::move(s));
            }
          } else {
            ClusterRecursively(my_cluster, depth + 1, process_leaf);
          }
        },
        gran);
  }

  auto recursively_sketch_wrapper(GraphH &G) {
    auto ids = parlay::tabulate(n_points_, [&](uint32_t i) { return i; });
    parlay::internal::timer t;

    std::cout << "Fanout scheme: ";
    for (size_t i = 0; i < FANOUT_SCHEME.size(); i++) {
      std::cout << FANOUT_SCHEME[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "leader method: random" << std::endl;

    t.start();

    std::cout << "Using fused cluster+leaf build." << std::endl;
    std::atomic<size_t> fused_leaf_count{0};
    std::atomic<size_t> fused_leaf_comparisons{0};

    auto fused_leaf = [&](Bucket leaf) {
      size_t s = leaf.size();
      fused_leaf_count.fetch_add(1, std::memory_order_relaxed);
      fused_leaf_comparisons.fetch_add(s * (s - 1), std::memory_order_relaxed);
      if (s > 1) RunLeaf(G, leaf);
    };

    ClusterRecursively(ids, 0, fused_leaf);

    SEED = parlay::hash64(SEED);
    LEAF_COMPARISONS += fused_leaf_comparisons.load();
    t.next("fused cluster+leaf time");

    std::cout << "Total leaves: " << fused_leaf_count.load() << std::endl;
    std::cout << "# of comparisons during leaf building: " << LEAF_COMPARISONS << std::endl;

    std::cout << "=== Leaf Building Stats ===" << std::endl;

    if (ASSIGN_SLOTS_.load(std::memory_order_relaxed) > 0) {
      const size_t emitted = ASSIGN_EMITTED_.load(std::memory_order_relaxed);
      const size_t slots = ASSIGN_SLOTS_.load(std::memory_order_relaxed);
      std::cout << "deep-assign emitted slots: " << emitted << " / " << slots
                << " (" << (100.0 * emitted / slots) << "%"
                << (ASSIGN_MARGIN_TAU > 0.0
                        ? ", margin_tau=" + std::to_string(ASSIGN_MARGIN_TAU)
                        : std::string())
                << ")" << std::endl;
    }
  }

  void multiple_clustertrees(GraphH &G, size_t n_points, long cluster_size,
                             long num_clusters) {
    n_points_ = n_points;
    G.allocate_graph(n_points);
    for (auto i = 0; i < num_clusters; i++) {
      recursively_sketch_wrapper(G);
      std::cout << "Built cluster " << i << " of " << num_clusters << std::endl;
      std::cout << "Leaf count: " << leaf_count << std::endl;
      std::cout << "# of comparisons during clustering: " << COMPARISONS
                << std::endl;
    }
  }

  void RunLeaf(GraphH &G,
               parlay::sequence<uint32_t> &active_indices) {
    leaf_count++;
    LeafKNN(G, active_indices);
    MaybeReportProgress();
  }

  void MaybeReportProgress() {
    if (progress_step_ == 0) return;
    const size_t done = leaves_done_.fetch_add(1, std::memory_order_relaxed) + 1;
    size_t expected = next_milestone_.load(std::memory_order_relaxed);
    if (done < expected) return;

    while (done >= expected) {
      if (next_milestone_.compare_exchange_weak(
              expected, expected + progress_step_,
              std::memory_order_relaxed)) {
        const double pct = 100.0 * static_cast<double>(done) /
                           static_cast<double>(progress_total_est_);
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(
            now - progress_start_).count();
        const double eta = (done > 0)
            ? elapsed * static_cast<double>(progress_total_est_ - done) /
                  static_cast<double>(done)
            : 0.0;
        char buf[192];
        std::snprintf(buf, sizeof(buf),
            "[progress] cluster+leaf ~%.0f%% (%zu/%zu est leaves)  "
            "elapsed=%.0fs  eta~%.0fs\n",
            pct, done, progress_total_est_, elapsed, eta);
        std::fputs(buf, stderr);
        return;
      }

    }
  }

  void SetProgressTarget(size_t est_total_leaves) {
    progress_total_est_ = est_total_leaves;

    progress_step_ = (est_total_leaves >= 20)
        ? est_total_leaves / 20
        : 0;
    leaves_done_.store(0, std::memory_order_relaxed);
    next_milestone_.store(progress_step_, std::memory_order_relaxed);
    progress_start_ = std::chrono::steady_clock::now();
  }

  void LeafKNN(GraphH &G,
                     parlay::sequence<uint32_t> &active_indices) {
    if (HAMMING_LEAF && rsqpr1_ && rsqpr_) {
      rsq_knn_hamming_rescore(G, *rsqpr1_, *rsqpr_,
                             absl::MakeSpan(active_indices), MST_DEG);
    } else if (rsqpr_) {
      rsq_knn(G, *rsqpr_, absl::MakeSpan(active_indices), MST_DEG);
    } else {
      std::cerr << "FATAL: leaf building requires the RSQ index." << std::endl;
      std::abort();
    }
  }

  size_t SEED = 555;
  double FRACTION_LEADERS = 0.005;
  size_t TOP_LEVEL_NUM_LEADERS = 1024;
  size_t MAX_NUM_LEADERS = 1500;
  size_t MAX_CLUSTER_SIZE = 512;
  size_t MIN_CLUSTER_SIZE = 100;

  std::atomic<size_t> leaves_done_{0};
  std::atomic<size_t> next_milestone_{0};
  size_t progress_total_est_ = 0;
  size_t progress_step_ = 0;
  std::chrono::steady_clock::time_point progress_start_;
  int MAX_DEPTH = 3;
  int CONCERNING_DEPTH = 2;
  double TOO_SMALL_SHRINKAGE_FRACTION = 0.8;
  size_t MST_DEG = 3;

  std::vector<int> FANOUT_SCHEME = {};
  std::atomic<size_t> COMPARISONS = 0;
  size_t LEAF_COMPARISONS = 0;

  pipnn::RsqPointRange* rsqpr_ = nullptr;

  pipnn::RsqPointRange* rsqpr1_ = nullptr;
  bool HAMMING_LEAF = false;
  bool HAMMING_ASSIGN = false;

  double ASSIGN_MARGIN_TAU = 0.0;

  std::atomic<size_t> ASSIGN_EMITTED_{0};
  std::atomic<size_t> ASSIGN_SLOTS_{0};

  bool CAPTURE_CELL0 = false;
  parlay::sequence<uint32_t> CELL0_;
  size_t n_points_ = 0;
};

template <typename Point, typename PointRange, typename indexType>
struct pipnn_index {
  using GraphI = Graph<indexType>;
  using PR = PointRange;

  pipnn_index() {}

  bool codes_cached_ = false;
  std::unique_ptr<pipnn::RsqPointRange> cached_rsqpr_;
  std::unique_ptr<pipnn::RsqPointRange> cached_rsqpr1_;

  std::unique_ptr<pipnn::RsqPointRange> init_rsq(PR& Points,
                                                               int rsq_bits) {
    parlay::internal::timer rsq_timer;
    rsq_timer.start();
    std::cout << "Building RsqPointRange for quantized scoring ("
              << rsq_bits << "-bit)..." << std::endl;
    size_t n = Points.size();
    size_t d = Points.dimension();
    std::vector<absl::Span<const float>> datapoints(n);
    parlay::parallel_for(0, n, [&](size_t i) {
      datapoints[i] = absl::MakeSpan(
          reinterpret_cast<const float*>(Points[i].data()), d);
    });
    rsq_timer.next("RSQ span creation time");

    auto scheme = (rsq_bits == 8) ? pipnn::QuantizationScheme::kEightBit
                                 : pipnn::QuantizationScheme::kOneBit;
    pipnn::RsqEncoder quantizer(d,42, scheme);
    auto base_points_or = pipnn::RsqBasePoints::Create(
        quantizer, absl::MakeSpan(datapoints),
true);
    rsq_timer.next("RSQ encoding time");
    auto rsqpr_or = [&]() -> absl::StatusOr<pipnn::RsqPointRange> {
      if (!base_points_or.ok()) return base_points_or.status();
      return pipnn::RsqPointRange(std::move(*base_points_or));
    }();
    if (rsqpr_or.ok()) {
      auto rsqpr_storage = std::make_unique<pipnn::RsqPointRange>(std::move(*rsqpr_or));
      std::cout << "RsqPointRange built: " << n << " points, " << d << " dims, "
                << rsq_bits << "-bit" << std::endl;
      return rsqpr_storage;
    } else {
      std::cerr << "WARNING: RSQ index construction failed" << std::endl;
      return nullptr;
    }
  }

  std::unique_ptr<pipnn::RsqPointRange>
  init_rsq_streaming(PR& Points, int rsq_bits) {
    parlay::internal::timer rsq_timer;
    rsq_timer.start();
    assert(Points.deferred_load &&
           "init_rsq_streaming requires deferred-load PointRange");
    if (!Points.deferred_src_is_float ||
        (Points.deferred_src_elem_bytes != 4 &&
         Points.deferred_src_elem_bytes != 2)) {
      std::cerr << "init_rsq_streaming: only f32/f16 sources supported "
                   "(got src_elem_bytes="
                << Points.deferred_src_elem_bytes
                << " is_float=" << Points.deferred_src_is_float << ")"
                << std::endl;
      return nullptr;
    }
    const bool src_is_f16 = (Points.deferred_src_elem_bytes == 2);
    const size_t n = Points.size();
    const size_t d = Points.dimension();
    const size_t src_row_bytes = Points.deferred_src_elem_bytes * d;
    std::cout << "init_rsq_streaming: n=" << n << " d=" << d
              << " rsq_bits=" << rsq_bits
              << " src=" << (src_is_f16 ? "f16" : "f32")
              << " src_row_bytes=" << src_row_bytes << std::endl;

    auto scheme = (rsq_bits == 8) ? pipnn::QuantizationScheme::kEightBit
                                 : pipnn::QuantizationScheme::kOneBit;
    pipnn::RsqEncoder quantizer(d,42, scheme);
    rsq_timer.next("RSQ quantizer construct");

    auto bp_or = pipnn::RsqBasePoints::Create(
        quantizer, n,true);
    if (!bp_or.ok()) {
      std::cerr << "init_rsq_streaming: empty Create() failed: "
                << bp_or.status() << std::endl;
      return nullptr;
    }
    auto bp = std::move(*bp_or);
    rsq_timer.next("RSQ alloc empty output");

    int fd = ::open(Points.deferred_path.c_str(), O_RDONLY);
    if (fd == -1) {
      std::cerr << "init_rsq_streaming: open() failed for "
                << Points.deferred_path << std::endl;
      return nullptr;
    }
    void* mp = ::mmap(nullptr, Points.deferred_file_size, PROT_READ,
                      MAP_PRIVATE, fd, 0);
    if (mp == MAP_FAILED) {
      ::close(fd);
      std::cerr << "init_rsq_streaming: mmap() failed" << std::endl;
      return nullptr;
    }
    char* file_data = static_cast<char*>(mp);
    ::madvise(file_data, Points.deferred_file_size, MADV_SEQUENTIAL);
    const char* data_start = file_data + Points.deferred_h5_offset;
    const long pg = ::sysconf(_SC_PAGESIZE);

    constexpr size_t CHUNK_POINTS = 65536;
    auto rss_kb = []() -> long {
      FILE* f = std::fopen("/proc/self/status", "r");
      if (!f) return -1;
      char buf[256];
      long rss = -1;
      while (std::fgets(buf, sizeof(buf), f)) {
        if (std::strncmp(buf, "VmRSS:", 6) == 0) {
          std::sscanf(buf + 6, "%ld", &rss);
          break;
        }
      }
      std::fclose(f);
      return rss;
    };
    long rss_start = rss_kb();
    std::cout << "init_rsq_streaming: encode start, VmRSS="
              << (rss_start >> 10) << " MiB" << std::endl;
    for (size_t base = 0; base < n; base += CHUNK_POINTS) {
      const size_t hi = std::min(base + CHUNK_POINTS, n);
      if (src_is_f16) {
        parlay::parallel_for(base, hi, [&](size_t i) {
          thread_local std::vector<float> tl_decode;
          if (tl_decode.size() < d) tl_decode.resize(d);
          float* dst = tl_decode.data();
          const uint16_t* src = reinterpret_cast<const uint16_t*>(
              data_start + i * src_row_bytes);
          unsigned int j = 0;
          for (; j + 8 <= d; j += 8) {
            __m128i h8 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(src + j));
            __m256 f8 = _mm256_cvtph_ps(h8);
            _mm256_storeu_ps(dst + j, f8);
          }
          if (j < d) {
            alignas(16) uint16_t tmp[8] = {0};
            std::memcpy(tmp, src + j, (d - j) * sizeof(uint16_t));
            __m256 f8 = _mm256_cvtph_ps(
                _mm_load_si128(reinterpret_cast<const __m128i*>(tmp)));
            alignas(32) float out[8];
            _mm256_store_ps(out, f8);
            for (unsigned int kk = 0; kk < d - j; ++kk) dst[j + kk] = out[kk];
          }
          auto coords = absl::MakeConstSpan(dst, d);
          auto status = bp.UpdatePoint(quantizer, coords, i);
          (void)status;
        }, 1000);
      } else {
        parlay::parallel_for(base, hi, [&](size_t i) {
          auto coords = absl::MakeConstSpan(
              reinterpret_cast<const float*>(data_start + i * src_row_bytes), d);
          auto status = bp.UpdatePoint(quantizer, coords, i);
          (void)status;
        }, 1000);
      }

      const char* chunk_lo = data_start + base * src_row_bytes;
      const char* chunk_hi = data_start + hi * src_row_bytes;
      uintptr_t lo_a = (reinterpret_cast<uintptr_t>(chunk_lo) + pg - 1) &
                       ~(static_cast<uintptr_t>(pg) - 1);
      uintptr_t hi_a = reinterpret_cast<uintptr_t>(chunk_hi) &
                       ~(static_cast<uintptr_t>(pg) - 1);
      if (hi_a > lo_a) {
        ::madvise(reinterpret_cast<void*>(lo_a),
                  static_cast<size_t>(hi_a - lo_a), MADV_DONTNEED);
      }

      if (((base / CHUNK_POINTS) % 8) == 0 || hi == n) {
        long rss = rss_kb();
        std::cout << "  chunk [" << base << "," << hi << ") VmRSS="
                  << (rss >> 10) << " MiB" << std::endl;
      }
    }
    rsq_timer.next("RSQ streaming encode");

    ::munmap(file_data, Points.deferred_file_size);
    ::close(fd);

    auto rsqpr_storage = std::make_unique<pipnn::RsqPointRange>(std::move(bp));
    std::cout << "RsqPointRange built (streaming): " << n
              << " points, " << d << " dims, " << rsq_bits << "-bit" << std::endl;
    return rsqpr_storage;
  }

  std::unique_ptr<pipnn::RsqPointRange>
  derive_rsq1_from_rsq8(const pipnn::RsqPointRange& rsqpr8) {
    parlay::internal::timer t;
    t.start();
    const auto& bp8 = rsqpr8.base_points;
    if (!bp8.is_8bit()) {
      std::cerr << "derive_rsq1_from_rsq8: source range is not 8-bit"
                << std::endl;
      return nullptr;
    }
    const size_t n = bp8.num_datapoints();
    const size_t d = bp8.dimensionality();
    pipnn::RsqEncoder q1(d,42,
                           pipnn::QuantizationScheme::kOneBit);
    auto bp1_or = pipnn::RsqBasePoints::Create(
        q1, n,true);
    if (!bp1_or.ok()) {
      std::cerr << "derive_rsq1_from_rsq8: Create() failed: " << bp1_or.status()
                << std::endl;
      return nullptr;
    }
    auto bp1 = std::move(*bp1_or);
    const size_t nbytes1 = (d + 7) / 8;
    const float inv_qnorm1 =
        1.0f / std::sqrt(static_cast<float>(d) * 16129.0f);
    parlay::parallel_for(0, n, [&](size_t i) {
      const int8_t* c8 =
          reinterpret_cast<const int8_t*>(bp8.GetQuantizedDataPtr(i));
      thread_local std::vector<uint8_t> code1;
      if (code1.size() < nbytes1) code1.resize(nbytes1);
      std::memset(code1.data(), 0, nbytes1);
      int64_t qsn8 = 0;
      for (size_t j = 0; j < d; ++j) {
        const int32_t v = c8[j];
        qsn8 += v * v;
        code1[j >> 3] |=
            static_cast<uint8_t>(static_cast<uint8_t>(v < 0) << (j & 7));
      }
      const float scale8 = bp8.norm_scaling_factor(i);
      const float norm = scale8 * std::sqrt(static_cast<float>(qsn8));
      const float scale1 = norm * inv_qnorm1;
      pipnn::RsqBasePointView view{
          absl::MakeConstSpan(code1.data(), nbytes1),
scale1};
      auto status = bp1.UpdatePoint(view, i);
      (void)status;
    },512);
    t.next("RSQ1 derive from RSQ8 sign bits");
    return std::make_unique<pipnn::RsqPointRange>(std::move(bp1));
  }

  void build_topk(GraphI &G, commandLine& P, PR &Points,
                  long cluster_rounds, long cluster_size, long MSTDeg,
                  long k, long top_level_leaders,
                  double fraction_leaders, std::string fanout_scheme,
                  bool rsq, int rsq_bits,
                  double* encode_seconds_out = nullptr,
                  double* build_seconds_out = nullptr,
                  double buildtime_in = 0.0) {
    using HeapG = NghDistGraphHeap<indexType>;

    std::cout << "build_topk: k=" << k
              << " cluster_rounds=" << cluster_rounds
              << " cluster_size=" << cluster_size
              << " MST_deg=" << MSTDeg << std::endl;

    const bool did_encode = !codes_cached_;
    parlay::internal::timer encode_timer;
    encode_timer.start();

    auto& rsqpr_storage = cached_rsqpr_;
    if (!rsqpr_storage && rsq) {
      if (Points.deferred_load) {
        rsqpr_storage = init_rsq_streaming(Points, rsq_bits);
      } else {
        rsqpr_storage = init_rsq(Points, rsq_bits);
      }
      if (!rsqpr_storage) {
        std::cerr << "FATAL: RSQ encode failed." << std::endl;
        std::abort();
      }
    }

    auto advise_huge = [](const uint8_t* data, size_t len) {
      if (data == nullptr || len < (2u << 20)) return;
      const uintptr_t pg = 4096;
      uintptr_t lo = (reinterpret_cast<uintptr_t>(data) + pg - 1) & ~(pg - 1);
      uintptr_t hi = (reinterpret_cast<uintptr_t>(data) + len) & ~(pg - 1);
      if (hi > lo)
        ::madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_HUGEPAGE);
    };
    if (rsqpr_storage) {
      advise_huge(rsqpr_storage->base_points.quantized_data(),
                  rsqpr_storage->base_points.quantized_data_size());
    }

    auto& rsqpr1_storage = cached_rsqpr1_;
    const bool hamming_leaf = P.getOption("-hamming_leaf");
    const bool hamming_assign = P.getOption("-hamming_assign");

    const bool build_rsq1_panel = P.getOption("-build_rsq1_panel");
    if ((hamming_leaf || hamming_assign || build_rsq1_panel) && rsq && rsqpr_storage && !rsqpr1_storage) {
      if (rsq_bits != 8) {
        std::cerr << "WARNING: -hamming_leaf rescores with 8-bit codes but "
                     "rsq_bits=" << rsq_bits << "; rescore precision reduced.\n";
      }

      if (rsq_bits == 8) {
        rsqpr1_storage = derive_rsq1_from_rsq8(*rsqpr_storage);
      }
      if (!rsqpr1_storage) {
        if (Points.deferred_load) {
          rsqpr1_storage = init_rsq_streaming(Points,1);
        } else {
          rsqpr1_storage = init_rsq(Points,1);
        }
      }
      if (!rsqpr1_storage) {
        std::cerr << "FATAL: -hamming_leaf set but 1-bit encode failed\n";
        std::abort();
      }
      std::cout << "build_topk: 1-bit RSQ range built for"
                << (hamming_leaf ? " hamming_leaf" : "")
                << (hamming_assign ? " hamming_assign" : "")
                << " (1-bit Hamming GEMM + 8-bit rescore)\n";
      advise_huge(rsqpr1_storage->base_points.quantized_data(),
                  rsqpr1_storage->base_points.quantized_data_size());
    }

    codes_cached_ = true;
    const double enc_s_local = did_encode ? encode_timer.next_time() : 0.0;
    if (encode_seconds_out) *encode_seconds_out = enc_s_local;
    if (!did_encode)
      std::cout << "build_topk: reusing cached RSQ index (encode skipped)"
                << std::endl;

    parlay::internal::timer build_timer;
    build_timer.start();

    HeapG heap;

    const long k_build_opt = P.getOptionLongValue("-k_build", 0);
    const size_t k_build =
        (k_build_opt > k) ? static_cast<size_t>(k_build_opt)
                          : static_cast<size_t>(k);
    if (k_build != static_cast<size_t>(k))
      std::cout << "build_topk: k_build=" << k_build
                << " (heap width; output still k=" << k << ")" << std::endl;
    heap.set_k_max(k_build);

    cluster<HeapG, Point, PointRange, indexType> C;

    if (char* lk = P.getOptionValue("-leaf_k")) {
      int lkv = std::atoi(lk);
      if (lkv < 1) {
        std::cerr << "build_topk: -leaf_k must be >= 1, got " << lkv
                  << std::endl;
        std::abort();
      }
      MSTDeg = lkv;
      std::cout << "build_topk: leaf_k overridden to " << MSTDeg
                << " (= per-leaf top-k)" << std::endl;
    }
    C.MST_DEG = MSTDeg;
    C.MAX_CLUSTER_SIZE = cluster_size;
    C.MIN_CLUSTER_SIZE = 100;
    C.TOP_LEVEL_NUM_LEADERS = top_level_leaders;
    C.FRACTION_LEADERS = fraction_leaders;
    if (rsqpr_storage) C.rsqpr_ = rsqpr_storage.get();
    if (rsqpr1_storage) {
      C.rsqpr1_ = rsqpr1_storage.get();
      C.HAMMING_LEAF = hamming_leaf;
      C.HAMMING_ASSIGN = hamming_assign;
    }
    if (char* mt = P.getOptionValue("-assign_margin_tau")) {
      C.ASSIGN_MARGIN_TAU = std::atof(mt);
      std::cout << "build_topk: assign_margin_tau=" << C.ASSIGN_MARGIN_TAU
                << " (inverted margin gate on deep Hamming assignment)"
                << std::endl;
    }
    if (fanout_scheme != "") {
      std::stringstream ss(fanout_scheme);
      std::string token;
      while (std::getline(ss, token, ',')) {
        C.FANOUT_SCHEME.push_back(std::stoi(token));
      }
    }

    size_t fanout_product = 1;
    for (int f : C.FANOUT_SCHEME) fanout_product *= static_cast<size_t>(f);
    const size_t avg_leaf = std::max<size_t>(1, cluster_size / 3);
    const size_t est_total_leaves =
        static_cast<size_t>(cluster_rounds) *
        fanout_product * Points.size() / avg_leaf;
    C.SetProgressTarget(est_total_leaves);
    std::cout << "build_topk: progress target ~" << est_total_leaves
              << " leaves (cluster_rounds=" << cluster_rounds
              << ", fanout_product=" << fanout_product
              << ", avg_leaf=" << avg_leaf << ")" << std::endl;

    const bool nnd_cell_order = P.getOption("-nnd_cell_order");
    C.CAPTURE_CELL0 = nnd_cell_order;

    parlay::internal::timer t;
    t.start();
    C.multiple_clustertrees(heap, Points.size(), cluster_size, cluster_rounds);
    t.next("build_topk: clustertrees");

    if (long nnd = P.getOptionLongValue("-nn_descent", 0);
        nnd > 0 && rsqpr_storage) {
      const size_t filter_c =
          static_cast<size_t>(P.getOptionLongValue("-nnd_filter", 32));

      const size_t expand_m =
          static_cast<size_t>(P.getOptionLongValue("-nnd_expand_m", 0));

      std::unique_ptr<parlay::sequence<uint32_t>> order;
      if (nnd_cell_order && C.CELL0_.size() == Points.size()) {
        order = std::make_unique<parlay::sequence<uint32_t>>(
            parlay::tabulate(Points.size(),
                             [](size_t i) { return (uint32_t)i; }));
        parlay::integer_sort_inplace(
            *order, [&](uint32_t u) { return C.CELL0_[u]; });
        t.next("build_topk: nnd cell order");
      }
      std::cout << "build_topk: nn_descent passes=" << nnd
                << " filter_c=" << filter_c
                << " expand_m=" << expand_m
                << (order ? " cell_order" : "")
                << (rsqpr1_storage ? "" : " (no 1-bit panel: unfiltered)")
                << std::endl;
      for (long pass = 0; pass < nnd; ++pass) {
        nn_descent_pass(heap, *rsqpr_storage, rsqpr1_storage.get(),
                        Points.size(), k_build, filter_c,
                        expand_m, order.get());
        t.next("build_topk: nn_descent pass");
      }
    }

    heap.finalize();
    t.next("build_topk: finalize heap");

    G.allocate_graph(k, Points.size());
    parlay::parallel_for(0, Points.size(), [&](size_t i) {
      auto ngh = heap.Neighbors(i);

      size_t deg = std::min(ngh.size(), static_cast<size_t>(k));
      indexType* dst = &G[i].begin()[-1];
      dst[0] = static_cast<indexType>(deg);
      for (size_t j = 0; j < deg; ++j) dst[j + 1] = ngh[j].first;
    });
    t.next("build_topk: copy to G");

    const double qt_local = build_timer.next_time();
    if (build_seconds_out) *build_seconds_out = qt_local;

    if (char* out_dir = P.getOptionValue("-output")) {
      const size_t N = Points.size();
      const long ncols = k + 1;
      std::vector<int32_t> hk(N * static_cast<size_t>(ncols));
      std::vector<float> hd(N * static_cast<size_t>(ncols));
      parlay::parallel_for(0, N, [&](size_t i) {
        hk[i * ncols + 0] = static_cast<int32_t>(i + 1);
        hd[i * ncols + 0] = 0.0f;
        auto ngh = heap.Neighbors(i);
        const size_t deg = ngh.size();
        for (long j = 0; j < k; ++j) {
          if (static_cast<size_t>(j) < deg) {
            hk[i * ncols + 1 + j] = static_cast<int32_t>(ngh[j].first) + 1;
            hd[i * ncols + 1 + j] = ngh[j].second;
          } else {
            hk[i * ncols + 1 + j] = static_cast<int32_t>(i + 1);
            hd[i * ncols + 1 + j] = std::numeric_limits<float>::max();
          }
        }
      });
      const std::string algo = P.getOptionValue("-algo", "PiPNN");
      const std::string dataset_name = P.getOptionValue("-dataset_name", "unknown");
      const std::string task_name = P.getOptionValue("-task_name", "task1");
      const std::string cfg_name = P.getOptionValue("-cfg_name", "cfg");
      const std::string cfg_params = P.getOptionValue("-cfg_params", "");
      const double bt = did_encode ? enc_s_local : buildtime_in;
      const std::string fname = std::string(out_dir) + "/" + algo + "-" +
                                dataset_name + "-" + cfg_name + ".h5";
      write_sisap_result(fname, hk.data(), hd.data(), N, ncols, algo,
                         dataset_name, task_name, cfg_params, bt, qt_local);
      t.next("build_topk: write SISAP result h5");
    }
  }

};

}
