// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <algorithm>
#include <atomic>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "hwy/highway.h"
#include <absl/types/span.h>

#include <x86intrin.h>

#include "quantization/rsq_point_range.h"
#include "quantization/pipnn_rsq.h"

namespace parlayANN {
inline void check_leaf_k(size_t leaf_k) {
  if (leaf_k < 3 || leaf_k > 15) {
    std::cerr << "FATAL: -leaf_k must be in [3,15] (got " << leaf_k
              << "). This build only ships the fused top-K leaf kernel used by "
                 "the SISAP 2026 Task 1 submission." << std::endl;
    std::abort();
  }
}

template<typename GraphH>
void rsq_knn(GraphH& G, const pipnn::RsqPointRange& rsqpr,
            absl::Span<const uint32_t> indices, size_t leaf_k) {
  const size_t n = indices.size();
  if (n < 2 || leaf_k == 0) return;

  static thread_local std::vector<int> int_indices;
  int_indices.assign(indices.begin(), indices.end());

  static thread_local pipnn::ProcessLeafWorkspace workspace;

  check_leaf_k(leaf_k);
  pipnn::ProcessLeafTopKFused(rsqpr.base_points,
                              absl::MakeSpan(int_indices), leaf_k, &workspace);

  static thread_local std::vector<std::pair<uint32_t, float>> fwd_buf;
  fwd_buf.resize(std::max<size_t>(leaf_k, 2));
  std::pair<uint32_t, float>* const fwd = fwd_buf.data();
  auto merge_row = [&](uint32_t global_i, const float* row_s,
                       const int32_t* row_i, size_t row_len) {
    size_t m = 0;
    for (size_t r = 0; r < row_len; ++r) {
      const int32_t local_neighbor = row_i[r];
      if (local_neighbor < 0) continue;
      fwd[m++] = {indices[local_neighbor], row_s[r]};
    }
    if (m == 0) return;
    G.merge_neighbors(global_i, absl::MakeSpan(fwd, m));
    for (size_t r = 0; r < m; ++r) {
      std::pair<uint32_t, float> rev = {global_i, fwd[r].second};
      G.merge_neighbors(fwd[r].first, absl::MakeSpan(&rev, 1));
    }
  };

  {
    const float* tk_s = workspace.fused_topk_scores.data();
    const int32_t* tk_i = workspace.fused_topk_indices.data();
    for (size_t i = 0; i < n; ++i) {
      merge_row(indices[i], tk_s + i * leaf_k, tk_i + i * leaf_k, leaf_k);
    }
  }

}

HWY_INLINE int32_t Rsq8CorrectedDotFast(const uint8_t* a8, const uint8_t* b8,
                                       size_t d) {
  const __m256i ones16 = _mm256_set1_epi16(1);
  __m256i acc = _mm256_setzero_si256();
  size_t k = 0;
  for (; k + 32 <= d; k += 32) {
    const __m256i av =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a8 + k));
    const __m256i bv =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b8 + k));
    const __m256i aabs = _mm256_abs_epi8(av);
    const __m256i bsgn = _mm256_sign_epi8(bv, av);
    const __m256i prod = _mm256_maddubs_epi16(aabs, bsgn);
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, ones16));
  }
  __m128i lo = _mm256_castsi256_si128(acc);
  __m128i hi = _mm256_extracti128_si256(acc, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_hadd_epi32(s, s);
  s = _mm_hadd_epi32(s, s);
  int32_t total = _mm_cvtsi128_si32(s);
  for (; k < d; ++k) {
    total += static_cast<int>(static_cast<int8_t>(a8[k])) *
             static_cast<int>(static_cast<int8_t>(b8[k]));
  }
  return total;
}

template<typename GraphH>
void rsq_knn_hamming_rescore(GraphH& G,
                            const pipnn::RsqPointRange& rsqpr1,
                            const pipnn::RsqPointRange& rsqpr8,
                            absl::Span<const uint32_t> indices, size_t leaf_k) {
  const size_t n = indices.size();
  if (n < 2 || leaf_k == 0) return;
  static thread_local std::vector<int> int_indices;
  int_indices.assign(indices.begin(), indices.end());
  static thread_local pipnn::ProcessLeafWorkspace workspace;

  check_leaf_k(leaf_k);
  pipnn::ProcessLeafTopKFused(rsqpr1.base_points,
                              absl::MakeSpan(int_indices), leaf_k, &workspace);

  const auto& bp8 = rsqpr8.base_points;
  size_t dbytes = bp8.num_bytes_per_datapoint();


  static thread_local std::vector<uint8_t> codes8;
  static thread_local std::vector<float> norms8;
  codes8.resize(n * dbytes);
  norms8.resize(n);

  constexpr size_t kGatherPf = 2;
  for (size_t i = 0; i < std::min(n, kGatherPf); ++i) {
    const char* p =
        reinterpret_cast<const char*>(bp8.GetQuantizedDataPtr(indices[i]));
    for (size_t l = 0; l < dbytes; l += 64) _mm_prefetch(p + l, _MM_HINT_T0);
  }
  for (size_t i = 0; i < n; ++i) {
    if (i + kGatherPf < n) {
      const char* p = reinterpret_cast<const char*>(
          bp8.GetQuantizedDataPtr(indices[i + kGatherPf]));
      for (size_t l = 0; l < dbytes; l += 64) _mm_prefetch(p + l, _MM_HINT_T0);
    }
    std::memcpy(codes8.data() + i * dbytes,
                bp8.GetQuantizedDataPtr(indices[i]), dbytes);
    norms8[i] = bp8.norm_scaling_factor(indices[i]);
  }

  auto rescore = [&](size_t local_i, size_t local_j) -> float {
    const int32_t cd =
        Rsq8CorrectedDotFast(codes8.data() + local_i * dbytes,
                            codes8.data() + local_j * dbytes, dbytes);
    return -static_cast<float>(cd) * norms8[local_i] * norms8[local_j];
  };

  static thread_local std::vector<std::pair<uint32_t, float>> edge_arena;
  static thread_local std::vector<uint32_t> edge_cnt;
  const size_t cap = 2 * leaf_k;
  if (edge_arena.size() < n * cap) edge_arena.resize(n * cap);
  edge_cnt.assign(n, 0);

  static thread_local std::vector<std::pair<uint32_t, float>> spill;
  spill.clear();
  static thread_local std::vector<uint32_t> spill_tgt;
  spill_tgt.clear();
  auto push_edge = [&](size_t local_tgt, uint32_t global_src, float dist) {
    uint32_t& c = edge_cnt[local_tgt];
    if (c < cap) {
      edge_arena[local_tgt * cap + c] = {global_src, dist};
      ++c;
    } else {
      spill.emplace_back(global_src, dist);
      spill_tgt.push_back(static_cast<uint32_t>(local_tgt));
    }
  };
  auto score_row = [&](size_t i, const int32_t* row_i, size_t row_len) {
    for (size_t r = 0; r < row_len; ++r) {
      const int32_t local_j = row_i[r];
      if (local_j < 0) continue;
      const float dist = rescore(i, static_cast<size_t>(local_j));
      push_edge(i, indices[local_j], dist);
      push_edge(static_cast<size_t>(local_j), indices[i], dist);
    }
  };

  {
    const int32_t* tk_i = workspace.fused_topk_indices.data();
    for (size_t i = 0; i < n; ++i) {
      score_row(i, tk_i + i * leaf_k, leaf_k);
    }
  }
  for (size_t i = 0; i < n; ++i) {
    const size_t m = edge_cnt[i];
    if (m == 0) continue;
    auto* edges = edge_arena.data() + i * cap;
    std::sort(edges, edges + m, [](const std::pair<uint32_t, float>& a,
                                   const std::pair<uint32_t, float>& b) {
      return a.second < b.second;
    });
    G.merge_neighbors(indices[i], absl::MakeSpan(edges, m));
  }
  for (size_t s = 0; s < spill.size(); ++s) {
    G.merge_neighbors(indices[spill_tgt[s]], absl::MakeSpan(&spill[s], 1));
  }
}

inline constexpr uint32_t kInvalidLeader = 0xFFFFFFFFu;

template <size_t kSurface, typename Bucket>
inline void Rsq8RescoreExtractLeaders(
    const pipnn::RsqBasePoints& bp8, size_t dbytes,
    const std::vector<int>& leader_indices, const Bucket& ids, size_t fanout,
    parlay::sequence<std::pair<uint32_t, uint32_t>>& flat,
    const std::vector<pipnn::TopKResult<kSurface>>& results,
    size_t block_start, size_t block_len, double margin_tau,
    std::atomic<size_t>* emitted) {
  const size_t nlead = leader_indices.size();
  assert(fanout <= 16);
  size_t emitted_local = 0;
  for (size_t j = 0; j < block_len; ++j) {
    const size_t out_idx = block_start + j;
    const uint32_t pg = ids[out_idx];
    const uint8_t* pcode = bp8.GetQuantizedDataPtr(pg);

    if (j + 1 < block_len) {
      const char* nxt = reinterpret_cast<const char*>(
          bp8.GetQuantizedDataPtr(ids[out_idx + 1]));
      for (size_t l = 0; l < dbytes; l += 64)
        _mm_prefetch(nxt + l, _MM_HINT_T0);
    }

    float best_s[16];
    int best_l[16];
    size_t nbest = 0;
    for (size_t c = 0; c < kSurface; ++c) {
      const int32_t L = results[j].indices[c];
      if (L < 0 || static_cast<size_t>(L) >= nlead) continue;
      const uint32_t lg = static_cast<uint32_t>(leader_indices[L]);
      const float score =
          static_cast<float>(Rsq8CorrectedDotFast(pcode,
              bp8.GetQuantizedDataPtr(lg), dbytes)) *
          bp8.norm_scaling_factor(lg);

      if (nbest < fanout) {
        size_t p = nbest++;
        while (p > 0 && best_s[p - 1] < score) {
          best_s[p] = best_s[p - 1]; best_l[p] = best_l[p - 1]; --p;
        }
        best_s[p] = score; best_l[p] = L;
      } else if (score > best_s[fanout - 1]) {
        size_t p = fanout - 1;
        while (p > 0 && best_s[p - 1] < score) {
          best_s[p] = best_s[p - 1]; best_l[p] = best_l[p - 1]; --p;
        }
        best_s[p] = score; best_l[p] = L;
      }
    }

    size_t nemit = nbest;
    if (margin_tau > 0.0 && nbest > 1) {
      const float cut = static_cast<float>(margin_tau) *
                        std::max(std::abs(best_s[0]), 1e-12f);
      size_t r = 1;
      while (r < nbest && (best_s[0] - best_s[r]) <= cut) ++r;
      nemit = r;
    }
    for (size_t fn = 0; fn < fanout; ++fn) {
      flat[out_idx * fanout + fn] =
          (fn < nemit) ? std::pair<uint32_t, uint32_t>{
                             static_cast<uint32_t>(best_l[fn]), pg}
                       : std::pair<uint32_t, uint32_t>{kInvalidLeader, 0};
    }
    emitted_local += nemit;
  }
  if (emitted)
    emitted->fetch_add(emitted_local, std::memory_order_relaxed);
}

inline uint32_t HammingDist1Bit(const uint8_t* a, const uint8_t* b,
                                size_t nbytes) {
  uint64_t acc = 0;
  size_t i = 0;
  for (; i + 8 <= nbytes; i += 8) {
    uint64_t x, y;
    std::memcpy(&x, a + i, 8);
    std::memcpy(&y, b + i, 8);
    acc += static_cast<uint64_t>(__builtin_popcountll(x ^ y));
  }
  for (; i < nbytes; ++i)
    acc += static_cast<uint32_t>(__builtin_popcount(a[i] ^ b[i]));
  return static_cast<uint32_t>(acc);
}

template<typename Bucket>
void rsq_assign_leaders_hamming_rescore(
    const pipnn::RsqPointRange& rsqpr1,
    const pipnn::RsqPointRange& rsqpr8,
    const Bucket& ids, const Bucket& leaders, size_t fanout,
    parlay::sequence<std::pair<uint32_t, uint32_t>>& flat, int depth,
    double margin_tau = 0.0, std::atomic<size_t>* emitted = nullptr) {
  constexpr size_t kSurface = 10;
  const size_t N = ids.size();
  std::vector<int> leader_indices(leaders.begin(), leaders.end());

  auto pretransposed = pipnn::PreTransposedRsqDataset::Build(
      rsqpr1.base_points, absl::MakeSpan(leader_indices));
  const auto& bp8 = rsqpr8.base_points;
  const size_t dbytes = bp8.num_bytes_per_datapoint();

  static constexpr size_t kBlockSize = 2048;
  parlay::blocked_for(0, N, kBlockSize,
      [&](size_t, size_t start, size_t end) {
        size_t len = end - start;
        thread_local std::vector<int> qi;
        qi.resize(len);
        for (size_t j = 0; j < len; ++j)
          qi[j] = static_cast<int>(ids[start + j]);
        thread_local std::vector<pipnn::TopKResult<kSurface>> results;
        thread_local pipnn::AssignToLeadersWorkspace workspace;
        pipnn::AssignToLeadersTopK<kSurface>(
            rsqpr1.base_points, absl::MakeSpan(qi.data(), len), pretransposed,
            results, &workspace);
        Rsq8RescoreExtractLeaders<kSurface>(
            bp8, dbytes, leader_indices, ids, fanout, flat, results, start,
            len, margin_tau, emitted);
      });
}

template <typename HeapG>
void nn_descent_pass(HeapG& heap,
                     const pipnn::RsqPointRange& rsqpr8,
                     const pipnn::RsqPointRange* rsqpr1,
                     size_t n, size_t k, size_t filter_c,
                     size_t expand_m = 0,
                     const parlay::sequence<uint32_t>* order = nullptr) {
  const auto& bp8 = rsqpr8.base_points;
  const size_t dbytes8 = bp8.num_bytes_per_datapoint();
  const uint8_t* codes1 = nullptr;
  size_t nbytes1 = 0, stride1 = 0;
  if (rsqpr1 != nullptr && filter_c > 0) {
    codes1 = rsqpr1->base_points.quantized_data();
    nbytes1 = rsqpr1->base_points.num_bytes_per_datapoint();
    stride1 = rsqpr1->base_points.stride();
  }
  const bool use_filter = (codes1 != nullptr);

  parlay::sequence<uint32_t> snap(n * k, kInvalidLeader);
  parlay::parallel_for(0, n, [&](size_t u) {
    auto ngh = heap.Neighbors(u);
    const size_t deg = std::min(ngh.size(), k);
    const size_t base = u * k;
    if (expand_m > 0) {
      thread_local std::vector<std::pair<float, uint32_t>> row;
      row.clear();
      for (size_t i = 0; i < deg; ++i)
        row.emplace_back(ngh[i].second, ngh[i].first);
      std::sort(row.begin(), row.end());
      for (size_t i = 0; i < row.size(); ++i) snap[base + i] = row[i].second;
    } else {
      for (size_t i = 0; i < deg; ++i) snap[base + i] = ngh[i].first;
    }
  });
  const size_t m_out = (expand_m > 0) ? std::min(expand_m, k) : k;

  parlay::parallel_for(0, n, [&](size_t r) {
    const size_t u = (order != nullptr) ? (*order)[r] : r;
    thread_local std::vector<uint32_t> cand;
    thread_local std::vector<std::pair<uint32_t, uint32_t>> ham;
    thread_local std::vector<std::pair<uint32_t, float>> fwd;
    cand.clear();

    constexpr uint32_t kHashSlots = 512;
    thread_local std::vector<uint32_t> ht(kHashSlots, kInvalidLeader);
    std::fill(ht.begin(), ht.end(), kInvalidLeader);
    auto insert_new = [&](uint32_t w) -> bool {
      uint32_t h = (w * 0x9E3779B9u) >> 23;
      while (true) {
        uint32_t& slot = ht[h & (kHashSlots - 1)];
        if (slot == kInvalidLeader) { slot = w; return true; }
        if (slot == w) return false;
        ++h;
      }
    };
    insert_new(static_cast<uint32_t>(u));
    const uint32_t* urow = snap.data() + u * k;

    for (size_t i = 0; i < k; ++i) {
      const uint32_t v = urow[i];
      if (v == kInvalidLeader) break;
      insert_new(v);
    }
    for (size_t i = 0; i < m_out; ++i) {
      const uint32_t v = urow[i];
      if (v == kInvalidLeader) break;
      const uint32_t* vrow = snap.data() + static_cast<size_t>(v) * k;
      for (size_t j = 0; j < k; ++j) {
        const uint32_t w = vrow[j];
        if (w == kInvalidLeader) break;
        if (insert_new(w)) cand.push_back(w);
      }
    }
    if (cand.empty()) return;

    constexpr size_t kPf = 8;
    if (use_filter && cand.size() > filter_c) {
      const uint8_t* ucode1 = codes1 + static_cast<size_t>(u) * stride1;
      ham.clear();
      ham.reserve(cand.size());
      for (size_t ci = 0; ci < cand.size(); ++ci) {
        if (ci + kPf < cand.size()) {
          const uint8_t* p =
              codes1 + static_cast<size_t>(cand[ci + kPf]) * stride1;
          _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<const char*>(p) + 64, _MM_HINT_T0);
        }
        const uint32_t w = cand[ci];
        ham.emplace_back(
            HammingDist1Bit(ucode1, codes1 + static_cast<size_t>(w) * stride1,
                            nbytes1),
            w);
      }
      std::nth_element(ham.begin(), ham.begin() + filter_c, ham.end());
      cand.clear();
      for (size_t i = 0; i < filter_c; ++i) cand.push_back(ham[i].second);
    }

    constexpr size_t kPf2Depth = 4;
    constexpr size_t kPf2Lines = 8;
    const uint8_t* ucode8 = bp8.GetQuantizedDataPtr(u);
    const float unorm = bp8.norm_scaling_factor(u);
    fwd.clear();
    fwd.reserve(cand.size());
    for (size_t ci = 0; ci < cand.size(); ++ci) {
      if (ci + kPf2Depth < cand.size()) {
        const char* p = reinterpret_cast<const char*>(
            bp8.GetQuantizedDataPtr(cand[ci + kPf2Depth]));
        for (size_t l = 0; l < kPf2Lines; ++l)
          _mm_prefetch(p + 64 * l, _MM_HINT_T0);
      }
      const uint32_t w = cand[ci];
      const int32_t cd =
          Rsq8CorrectedDotFast(ucode8, bp8.GetQuantizedDataPtr(w), dbytes8);
      const float dist =
          -static_cast<float>(cd) * unorm * bp8.norm_scaling_factor(w);
      fwd.emplace_back(w, dist);
    }
    std::sort(fwd.begin(), fwd.end(),
              [](const std::pair<uint32_t, float>& a,
                 const std::pair<uint32_t, float>& b) {
                return a.second < b.second;
              });
    heap.merge_neighbors(u, absl::MakeSpan(fwd.data(), fwd.size()));
    for (const auto& e : fwd) {
      std::pair<uint32_t, float> rev = {static_cast<uint32_t>(u), e.second};
      heap.merge_neighbors(e.first, absl::MakeSpan(&rev, 1));
    }
  },256);
}

template<typename Bucket>
void rsq_assign_leaders(const pipnn::RsqPointRange& rsqpr,
                       const Bucket& ids, const Bucket& leaders,
                       size_t fanout,
                       parlay::sequence<std::pair<uint32_t, uint32_t>>& flat,
                       int depth = -1) {
  const size_t N = ids.size();

  std::vector<int> leader_indices(leaders.begin(), leaders.end());

  auto pretransposed = pipnn::PreTransposedRsqDataset::Build(
      rsqpr.base_points, absl::MakeSpan(leader_indices));

  auto extract_block = [&](auto& results, size_t block_start, size_t block_len) {
    for (size_t j = 0; j < block_len; ++j) {
      size_t out_idx = block_start + j;
      for (size_t fn = 0; fn < fanout; ++fn) {
        int32_t leader = results[j].indices[fn];
        if (leader >= 0) {
          flat[out_idx * fanout + fn] = {static_cast<uint32_t>(leader), ids[out_idx]};
        }
      }
    }
  };

  if (depth == 0 && fanout > 1) {
    static constexpr size_t kBlockSize = 1024;
    if (fanout <= 3) {
      parlay::blocked_for(0, N, kBlockSize,
          [&](size_t, size_t start, size_t end) {
            thread_local std::vector<pipnn::TopKResult<3>> results;
            thread_local pipnn::AssignToLeadersWorkspace workspace;
            pipnn::AssignToLeadersConsecutiveTopK<3>(
                rsqpr.base_points, start, end, pretransposed,
                results, &workspace);
            extract_block(results, start, end - start);
          });
    } else {
      parlay::blocked_for(0, N, kBlockSize,
          [&](size_t, size_t start, size_t end) {
            thread_local std::vector<pipnn::TopKResult<10>> results;
            thread_local pipnn::AssignToLeadersWorkspace workspace;
            pipnn::AssignToLeadersConsecutiveTopK<10>(
                rsqpr.base_points, start, end, pretransposed,
                results, &workspace);
            extract_block(results, start, end - start);
          });
    }
  } else {
    static constexpr size_t kBlockSize = 2048;
    if (fanout <= 1) {
      parlay::blocked_for(0, N, kBlockSize,
          [&](size_t, size_t start, size_t end) {
            size_t len = end - start;
            thread_local std::vector<int> qi;
            qi.resize(len);
            for (size_t j = 0; j < len; ++j)
              qi[j] = static_cast<int>(ids[start + j]);
            thread_local std::vector<pipnn::TopKResult<1>> results;
            thread_local pipnn::AssignToLeadersWorkspace workspace;
            pipnn::AssignToLeadersTop1(
                rsqpr.base_points, absl::MakeSpan(qi.data(), len), pretransposed,
                results, &workspace);
            extract_block(results, start, len);
          });
    } else if (fanout <= 3) {
      parlay::blocked_for(0, N, kBlockSize,
          [&](size_t, size_t start, size_t end) {
            size_t len = end - start;
            thread_local std::vector<int> qi;
            qi.resize(len);
            for (size_t j = 0; j < len; ++j)
              qi[j] = static_cast<int>(ids[start + j]);
            thread_local std::vector<pipnn::TopKResult<3>> results;
            thread_local pipnn::AssignToLeadersWorkspace workspace;
            pipnn::AssignToLeadersTopK<3>(
                rsqpr.base_points, absl::MakeSpan(qi.data(), len), pretransposed,
                results, &workspace);
            extract_block(results, start, len);
          });
    } else {
      parlay::blocked_for(0, N, kBlockSize,
          [&](size_t, size_t start, size_t end) {
            size_t len = end - start;
            thread_local std::vector<int> qi;
            qi.resize(len);
            for (size_t j = 0; j < len; ++j)
              qi[j] = static_cast<int>(ids[start + j]);
            thread_local std::vector<pipnn::TopKResult<10>> results;
            thread_local pipnn::AssignToLeadersWorkspace workspace;
            pipnn::AssignToLeadersTopK<10>(
                rsqpr.base_points, absl::MakeSpan(qi.data(), len), pretransposed,
                results, &workspace);
            extract_block(results, start, len);
          });
    }
  }
}

}
