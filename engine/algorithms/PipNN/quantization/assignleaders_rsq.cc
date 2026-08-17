#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "rsq_encoder.h"
#include "absl/base/prefetch.h"
#include "pipnn_rsq.h"
#include "absl/types/span.h"
#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE

#define HWY_TARGET_INCLUDE \
  "assignleaders_rsq.cc"

#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace pipnn {
namespace HWY_NAMESPACE {
namespace {
namespace hn = hwy::HWY_NAMESPACE;

#include "pipnn_rsq-inl.h"

template <size_t K>
HWY_INLINE void InsertTopK(TopKResult<K>& r, float score, int32_t idx) {
  if (score >= r.scores[K - 1]) return;
  size_t pos = K - 1;
  while (pos > 0 && score < r.scores[pos - 1]) {
    r.scores[pos] = r.scores[pos - 1];
    r.indices[pos] = r.indices[pos - 1];
    --pos;
  }
  r.scores[pos] = score;
  r.indices[pos] = idx;
}

void DecodeGatheredQueries(
    const RsqBasePoints& quantized_points,
    absl::Span<const int> query_indices, AssignToLeadersWorkspace* workspace) {
  using Int8 = hn::ScalableTag<int8_t>;
  using Uint8 = hn::ScalableTag<uint8_t>;
  using Int32 = hn::ScalableTag<int32_t>;

  const Int8 i8;
  const Uint8 u8;
  const Int32 i32;

  const bool is_8bit = quantized_points.is_8bit();
  const size_t N = hn::Lanes(i8);
  const size_t num_queries = query_indices.size();
  const size_t bytes_per_point = quantized_points.num_bytes_per_datapoint();
  const size_t stride = quantized_points.stride();
  const size_t num_dim_chunks = hwy::DivCeil(bytes_per_point, N);
  const size_t decoded_dim = is_8bit ? num_dim_chunks * N
                                     : num_dim_chunks * 8 * N;

  const uint8_t* raw = quantized_points.quantized_data();

  workspace->all_q_decoded.resize(num_queries * decoded_dim);
  workspace->all_q_byte_sums.resize(num_queries);
  workspace->q_norms.resize(num_queries);

  if (is_8bit) {
    DecodeQueries8BitWithSums(u8, i8, i32, raw, stride, num_queries,
                              bytes_per_point, num_dim_chunks, N,
                              workspace->all_q_decoded.data(),
                              workspace->all_q_byte_sums.data(),
                              query_indices.data());
  }

  for (size_t i = 0; i < num_queries; ++i) {
    const int idx = query_indices[i];
    workspace->q_norms[i] = quantized_points.norm_scaling_factor(idx);
  }
}

void DecodeConsecutiveQueries(
    const RsqBasePoints& quantized_points, size_t query_begin,
    size_t query_end, AssignToLeadersWorkspace* workspace) {
  using Int8 = hn::ScalableTag<int8_t>;
  using Uint8 = hn::ScalableTag<uint8_t>;
  using Int32 = hn::ScalableTag<int32_t>;

  const Int8 i8;
  const Uint8 u8;
  const Int32 i32;

  const bool is_8bit = quantized_points.is_8bit();
  const size_t N = hn::Lanes(i8);
  const size_t num_queries = query_end - query_begin;
  const size_t bytes_per_point = quantized_points.num_bytes_per_datapoint();
  const size_t stride = quantized_points.stride();
  const size_t num_dim_chunks = hwy::DivCeil(bytes_per_point, N);
  const size_t decoded_dim = is_8bit ? num_dim_chunks * N
                                     : num_dim_chunks * 8 * N;

  const uint8_t* q_data =
      quantized_points.quantized_data() + query_begin * stride;

  workspace->all_q_decoded.resize(num_queries * decoded_dim);
  workspace->q_norms.resize(num_queries);
  workspace->all_q_byte_sums.resize(num_queries);

  if (is_8bit) {
    DecodeQueries8BitWithSums(u8, i8, i32, q_data, stride, num_queries,
                              bytes_per_point, num_dim_chunks, N,
                              workspace->all_q_decoded.data(),
                              workspace->all_q_byte_sums.data());
  }

  for (size_t i = 0; i < num_queries; ++i) {
    workspace->q_norms[i] =
        quantized_points.norm_scaling_factor(query_begin + i);
  }
}

void AssignToLeadersTop1Impl(
    const RsqBasePoints& quantized_points,
    absl::Span<const int> query_indices,
    const PreTransposedRsqDataset& leaders,
    std::vector<TopKResult<1>>& top1_results,
    AssignToLeadersWorkspace* workspace) {
  static constexpr size_t kMqTop1 = 6;

  using Int8 = hn::ScalableTag<int8_t>;
  using Uint8 = hn::ScalableTag<uint8_t>;
  using Int32 = hn::ScalableTag<int32_t>;
  using Float = hn::ScalableTag<float>;

  const Int8 i8;
  const Uint8 u8;
  const Int32 i32;
  const Float f32;

  const bool is_8bit = quantized_points.is_8bit();
  const bool is_1bit = quantized_points.is_1bit();
  const bool use_hamming = is_1bit;
  const size_t N = hn::Lanes(i8);
  const size_t kPoints = hn::Lanes(i32);
  const size_t NF = hn::Lanes(f32);
  const size_t num_queries = query_indices.size();
  const size_t bytes_per_point = quantized_points.num_bytes_per_datapoint();
  const size_t dims = quantized_points.dimensionality();
  const size_t stride = quantized_points.stride();
  const size_t num_dim_chunks = hwy::DivCeil(bytes_per_point, N);
  const size_t num_hamming_tiles = hwy::DivCeil(bytes_per_point, 4);
  const size_t total_tiles = use_hamming ? num_hamming_tiles
                           : is_8bit ? num_dim_chunks * (N / 4)
                                     : num_dim_chunks * (2 * N);
  const size_t decoded_dim = is_8bit ? num_dim_chunks * N
                                     : num_dim_chunks * 8 * N;

  workspace->q_norms.resize(num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    workspace->q_norms[i] =
        quantized_points.norm_scaling_factor(query_indices[i]);
  }

  const uint8_t* raw = quantized_points.quantized_data();

  workspace->all_q_decoded.resize(kMqTop1 * decoded_dim);
  workspace->all_q_byte_sums.resize(kMqTop1);

  const size_t num_panels = leaders.num_panels();
  const size_t panel_bytes = leaders.panel_bytes();
  const float inf = std::numeric_limits<float>::max();

  top1_results.assign(num_queries, TopKResult<1>{});

  static constexpr size_t kL2Budget = 768 * 1024;
  const size_t panels_per_block =
      std::max(size_t{4}, (kL2Budget / panel_bytes) & ~size_t{3});

  const size_t min_size = num_queries * NF;
  workspace->min_scores.resize(min_size);
  workspace->min_indices.resize(min_size);
  std::fill(workspace->min_scores.begin(), workspace->min_scores.end(), inf);
  std::fill(workspace->min_indices.begin(), workspace->min_indices.end(),
            int32_t{-1});

  const auto lane_idx = hn::Iota(i32, 0);

  for (size_t p_block = 0; p_block < num_panels; p_block += panels_per_block) {
    const size_t p_end = std::min(p_block + panels_per_block, num_panels);

    for (size_t q_start = 0; q_start < num_queries; q_start += kMqTop1) {
      const size_t actual_mq = std::min(kMqTop1, num_queries - q_start);

      const int* batch_indices = query_indices.data() + q_start;
      const int8_t* query_ptrs[kMqTop1];
      const uint8_t* hamming_query_ptrs[kMqTop1];
      float q_norms[kMqTop1];

      if (use_hamming) {
        for (size_t q = 0; q < actual_mq; ++q) {
          hamming_query_ptrs[q] = raw + batch_indices[q] * stride;
          q_norms[q] = workspace->q_norms[q_start + q];
        }
      } else if (is_8bit) {
        DecodeQueries8BitWithSums(u8, i8, i32, raw, stride, actual_mq,
                                  bytes_per_point, num_dim_chunks, N,
                                  workspace->all_q_decoded.data(),
                                  workspace->all_q_byte_sums.data(),
                                  batch_indices);
        for (size_t q = 0; q < actual_mq; ++q) {
          query_ptrs[q] = workspace->all_q_decoded.data() + q * decoded_dim;
          q_norms[q] = workspace->q_norms[q_start + q];
        }
      }

      for (size_t q = actual_mq; q < kMqTop1; ++q) {
        if (use_hamming) {
          hamming_query_ptrs[q] = hamming_query_ptrs[actual_mq - 1];
        } else {
          query_ptrs[q] = query_ptrs[actual_mq - 1];
        }
        q_norms[q] = 0.0f;
      }

      const size_t qbuf_tile_stride = num_hamming_tiles * N;
      HWY_ALIGN uint8_t hamming_qbuf[kMqTop1 * 64 * 64];
      if (use_hamming) {
        PreBroadcastHammingQueries<kMqTop1>(u8, i32, hamming_query_ptrs,
            num_hamming_tiles, bytes_per_point, N, hamming_qbuf);
      }

      hn::Vec<Float> neg_corr[kMqTop1];
      if (!use_hamming) {
        for (size_t q = 0; q < actual_mq; ++q) {
          neg_corr[q] =
              hn::Set(f32, 128.0f * workspace->all_q_byte_sums[q]);
        }
      }
      for (size_t q = use_hamming ? 0 : actual_mq; q < kMqTop1; ++q) {
        neg_corr[q] = hn::Zero(f32);
      }

      hn::Vec<Float> min_v[kMqTop1];
      hn::Vec<Int32> min_idx_v[kMqTop1];
      for (size_t q = 0; q < actual_mq; ++q) {
        min_v[q] =
            hn::Load(f32, workspace->min_scores.data() + (q_start + q) * NF);
        min_idx_v[q] =
            hn::Load(i32, workspace->min_indices.data() + (q_start + q) * NF);
      }
      for (size_t q = actual_mq; q < kMqTop1; ++q) {
        min_v[q] = hn::Set(f32, std::numeric_limits<float>::max());
        min_idx_v[q] = hn::Set(i32, -1);
      }

      const size_t num_db_points = leaders.num_points();

      const auto update_top1 = [&](size_t q, hn::Vec<Float> s,
                                   size_t panel_base) HWY_ATTR {
        const auto base_v =
            hn::Add(hn::Set(i32, static_cast<int32_t>(panel_base)), lane_idx);
        if (HWY_UNLIKELY(panel_base + kPoints > num_db_points)) {
          const auto valid = hn::RebindMask(
              f32, hn::Lt(base_v,
                          hn::Set(i32, static_cast<int32_t>(num_db_points))));
          s = hn::IfThenElse(valid, s,
                             hn::Set(f32, std::numeric_limits<float>::max()));
        }
        const auto lt = hn::Lt(s, min_v[q]);
        min_v[q] = hn::IfThenElse(lt, s, min_v[q]);
        min_idx_v[q] =
            hn::IfThenElse(hn::RebindMask(i32, lt), base_v, min_idx_v[q]);
      };

      auto acc_to_dot = [&](hn::Vec<Int32>& a, size_t q) -> hn::Vec<Float> {
        if (use_hamming) {
          const auto h_f = hn::ConvertTo(f32, a);
          return hn::MulAdd(hn::Set(f32, -2.0f * 16129.0f), h_f,
                            hn::Set(f32, 16129.0f * static_cast<float>(dims)));
        } else {
          return hn::Sub(hn::ConvertTo(f32, a), neg_corr[q]);
        }
      };

      size_t p = p_block;

      for (; p + 4 <= p_end; p += 4) {
        const size_t base0 = p * kPoints;
        const size_t base1 = base0 + kPoints;
        const size_t base2 = base1 + kPoints;
        const size_t base3 = base2 + kPoints;

        hn::Vec<Int32> acc0[kMqTop1], acc1[kMqTop1], acc2[kMqTop1],
            acc3[kMqTop1];
        if (use_hamming) {
          HammingMicroKernelAccumulate4<kMqTop1>(
              u8, i8, i32, hamming_qbuf, qbuf_tile_stride,
              leaders.panel_data() + p * panel_bytes,
              leaders.panel_data() + (p + 1) * panel_bytes,
              leaders.panel_data() + (p + 2) * panel_bytes,
              leaders.panel_data() + (p + 3) * panel_bytes,
              num_hamming_tiles, N, acc0, acc1, acc2, acc3);
        } else {
          MicroKernelAccumulate4<kMqTop1>(
              u8, i8, i32, query_ptrs,
              leaders.panel_data() + p * panel_bytes,
              leaders.panel_data() + (p + 1) * panel_bytes,
              leaders.panel_data() + (p + 2) * panel_bytes,
              leaders.panel_data() + (p + 3) * panel_bytes, total_tiles, N, acc0,
              acc1, acc2, acc3);
        }

        const auto norms0 = hn::Neg(hn::Load(f32, leaders.norms() + base0));
        const auto norms1 = hn::Neg(hn::Load(f32, leaders.norms() + base1));
        const auto norms2 = hn::Neg(hn::Load(f32, leaders.norms() + base2));
        const auto norms3 = hn::Neg(hn::Load(f32, leaders.norms() + base3));
        for (size_t q = 0; q < actual_mq; ++q) {
          const auto d0 = hn::Mul(acc_to_dot(acc0[q], q), norms0);
          const auto d1 = hn::Mul(acc_to_dot(acc1[q], q), norms1);
          const auto d2 = hn::Mul(acc_to_dot(acc2[q], q), norms2);
          const auto d3 = hn::Mul(acc_to_dot(acc3[q], q), norms3);
          update_top1(q, d0, base0);
          update_top1(q, d1, base1);
          update_top1(q, d2, base2);
          update_top1(q, d3, base3);
        }
      }

      for (; p + 2 <= p_end; p += 2) {
        const size_t base0 = p * kPoints;
        const size_t base1 = base0 + kPoints;

        hn::Vec<Int32> acc0[kMqTop1], acc1[kMqTop1];
        if (use_hamming) {
          HammingMicroKernelAccumulate<kMqTop1>(
              u8, i8, i32, hamming_qbuf, qbuf_tile_stride,
              leaders.panel_data() + p * panel_bytes,
              leaders.panel_data() + (p + 1) * panel_bytes,
              num_hamming_tiles, N, acc0, acc1);
        } else {
          MicroKernelAccumulate<kMqTop1>(
              u8, i8, i32, query_ptrs,
              leaders.panel_data() + p * panel_bytes,
              leaders.panel_data() + (p + 1) * panel_bytes, total_tiles, N, acc0,
              acc1);
        }

        const auto norms0 = hn::Neg(hn::Load(f32, leaders.norms() + base0));
        const auto norms1 = hn::Neg(hn::Load(f32, leaders.norms() + base1));
        for (size_t q = 0; q < actual_mq; ++q) {
          const auto d0 = hn::Mul(acc_to_dot(acc0[q], q), norms0);
          const auto d1 = hn::Mul(acc_to_dot(acc1[q], q), norms1);
          update_top1(q, d0, base0);
          update_top1(q, d1, base1);
        }
      }

      for (; p < p_end; ++p) {
        const size_t base = p * kPoints;

        hn::Vec<Int32> acc[kMqTop1];
        if (use_hamming) {
          HammingMicroKernelAccumulate1<kMqTop1>(
              u8, i8, i32, hamming_qbuf, qbuf_tile_stride,
              leaders.panel_data() + p * panel_bytes,
              num_hamming_tiles, N, acc);
        } else {
          MicroKernelAccumulate1<kMqTop1>(u8, i8, i32, query_ptrs,
                                          leaders.panel_data() + p * panel_bytes,
                                          total_tiles, N, acc);
        }

        const auto norms_v = hn::Neg(hn::Load(f32, leaders.norms() + base));
        for (size_t q = 0; q < actual_mq; ++q) {
          const auto d = hn::Mul(acc_to_dot(acc[q], q), norms_v);
          update_top1(q, d, base);
        }
      }

      for (size_t q = 0; q < actual_mq; ++q) {
        hn::Store(min_v[q], f32,
                  workspace->min_scores.data() + (q_start + q) * NF);
        hn::Store(min_idx_v[q], i32,
                  workspace->min_indices.data() + (q_start + q) * NF);
      }
    }
  }

  for (size_t qi = 0; qi < num_queries; ++qi) {
    const auto mv = hn::Load(f32, workspace->min_scores.data() + qi * NF);
    const float min_val = hn::ReduceMin(f32, mv);
    if (min_val >= inf) continue;

    const auto eq = hn::Eq(mv, hn::Set(f32, min_val));
    const intptr_t lane = hn::FindKnownFirstTrue(f32, eq);
    const auto idx_v = hn::Load(i32, workspace->min_indices.data() + qi * NF);
    const int32_t idx = hn::ExtractLane(idx_v, lane);

    const float qn = workspace->q_norms[qi];
    top1_results[qi].scores[0] = min_val * qn;
    top1_results[qi].indices[0] = idx;
  }

}

template <size_t K>
void AssignToLeadersTopKImpl(
    const RsqBasePoints& quantized_points,
    absl::Span<const int> query_indices,
    const PreTransposedRsqDataset& leaders,
    std::vector<TopKResult<K>>& topk_results,
    AssignToLeadersWorkspace* workspace) {
  static constexpr size_t kMqK = 6;

  using Int8 = hn::ScalableTag<int8_t>;
  using Uint8 = hn::ScalableTag<uint8_t>;
  using Int32 = hn::ScalableTag<int32_t>;
  using Float = hn::ScalableTag<float>;

  const Int8 i8;
  const Uint8 u8;
  const Int32 i32;
  const Float f32;

  const bool is_8bit = quantized_points.is_8bit();
  const bool is_1bit = quantized_points.is_1bit();
  const bool use_hamming = is_1bit;
  const size_t N = hn::Lanes(i8);
  const size_t kPoints = hn::Lanes(i32);
  const size_t num_queries = query_indices.size();
  const size_t bytes_per_point = quantized_points.num_bytes_per_datapoint();
  const size_t dims = quantized_points.dimensionality();
  const size_t num_dim_chunks = hwy::DivCeil(bytes_per_point, N);
  const size_t num_hamming_tiles = hwy::DivCeil(bytes_per_point, 4);
  const size_t total_tiles = use_hamming ? num_hamming_tiles
                           : is_8bit ? num_dim_chunks * (N / 4)
                                     : num_dim_chunks * (2 * N);
  const size_t decoded_dim = is_8bit ? num_dim_chunks * N
                                     : num_dim_chunks * 8 * N;

  const size_t stride = quantized_points.stride();

  workspace->q_norms.resize(num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    workspace->q_norms[i] =
        quantized_points.norm_scaling_factor(query_indices[i]);
  }

  const uint8_t* raw = quantized_points.quantized_data();

  const size_t num_panels = leaders.num_panels();
  const size_t panel_bytes = leaders.panel_bytes();

  topk_results.assign(num_queries, TopKResult<K>{});

  const size_t NF = hn::Lanes(f32);
  const float inf = std::numeric_limits<float>::max();

  const auto lane_idx = hn::Iota(i32, 0);

  static constexpr size_t kL2Budget = 768 * 1024;
  const size_t panels_per_block =
      std::max(size_t{4}, (kL2Budget / panel_bytes) & ~size_t{3});
  const bool panels_fit_l2 = (num_panels * panel_bytes) <= kL2Budget;

  if (panels_fit_l2) {
    workspace->all_q_decoded.resize(kMqK * decoded_dim);
    workspace->all_q_byte_sums.resize(kMqK);

    for (size_t q_start = 0; q_start < num_queries; q_start += kMqK) {
      const size_t actual_mq = std::min(kMqK, num_queries - q_start);

      const int* batch_indices = query_indices.data() + q_start;
      if (!use_hamming) {
        if (is_8bit) {
          DecodeQueries8BitWithSums(u8, i8, i32, raw, stride, actual_mq,
                                    bytes_per_point, num_dim_chunks, N,
                                    workspace->all_q_decoded.data(),
                                    workspace->all_q_byte_sums.data(),
                                    batch_indices);
        }
      }

      const int8_t* query_ptrs[kMqK];
      const uint8_t* hamming_query_ptrs[kMqK];
      float q_norms[kMqK];
      for (size_t q = 0; q < actual_mq; ++q) {
        if (use_hamming) {
          hamming_query_ptrs[q] = raw + batch_indices[q] * stride;
        } else {
          query_ptrs[q] = workspace->all_q_decoded.data() + q * decoded_dim;
        }
        q_norms[q] = workspace->q_norms[q_start + q];
      }
      for (size_t q = actual_mq; q < kMqK; ++q) {
        if (use_hamming) {
          hamming_query_ptrs[q] = hamming_query_ptrs[actual_mq - 1];
        } else {
          query_ptrs[q] = query_ptrs[actual_mq - 1];
        }
        q_norms[q] = 0.0f;
      }

      const size_t qbuf_tile_stride_k = num_hamming_tiles * N;
      HWY_ALIGN uint8_t hamming_qbuf_k[kMqK * 64 * 64];
      if (use_hamming) {
        PreBroadcastHammingQueries<kMqK>(u8, i32, hamming_query_ptrs,
            num_hamming_tiles, bytes_per_point, N, hamming_qbuf_k);
      }

      hn::Vec<Float> neg_corr[kMqK];
      if (!use_hamming) {
        for (size_t q = 0; q < actual_mq; ++q) {
          neg_corr[q] =
              hn::Set(f32, 128.0f * workspace->all_q_byte_sums[q]);
        }
      }
      for (size_t q = use_hamming ? 0 : actual_mq; q < kMqK; ++q) {
        neg_corr[q] = hn::Zero(f32);
      }

      auto acc_to_dot = [&](hn::Vec<Int32>& a, size_t q) -> hn::Vec<Float> {
        if (use_hamming) {
          const auto h_f = hn::ConvertTo(f32, a);
          return hn::MulAdd(hn::Set(f32, -2.0f * 16129.0f), h_f,
                            hn::Set(f32, 16129.0f * static_cast<float>(dims)));
        } else {
          return hn::Sub(hn::ConvertTo(f32, a), neg_corr[q]);
        }
      };

      const auto inf_v = hn::Set(f32, inf);
      hn::Vec<Float> top_v[kMqK][K];
      hn::Vec<Int32> top_idx[kMqK][K];
      for (size_t q = 0; q < kMqK; ++q) {
        for (size_t k = 0; k < K; ++k) {
          top_v[q][k] = inf_v;
          top_idx[q][k] = hn::Set(i32, -1);
        }
      }

      const size_t num_db_points = leaders.num_points();
      const auto update_topk = [&](size_t q, hn::Vec<Float> s,
                                   size_t panel_base) HWY_ATTR {
        const auto base_v =
            hn::Add(hn::Set(i32, static_cast<int32_t>(panel_base)), lane_idx);
        if (HWY_UNLIKELY(panel_base + kPoints > num_db_points)) {
          const auto valid = hn::RebindMask(
              f32, hn::Lt(base_v,
                          hn::Set(i32, static_cast<int32_t>(num_db_points))));
          s = hn::IfThenElse(valid, s, hn::Set(f32, inf));
        }
        auto displaced = s;
        auto displaced_idx = base_v;
        for (size_t k = 0; k < K; ++k) {
          const auto lt = hn::Lt(displaced, top_v[q][k]);
          const auto lt_i = hn::RebindMask(i32, lt);
          const auto new_k = hn::IfThenElse(lt, displaced, top_v[q][k]);
          const auto new_k_idx =
              hn::IfThenElse(lt_i, displaced_idx, top_idx[q][k]);
          displaced = hn::IfThenElse(lt, top_v[q][k], displaced);
          displaced_idx = hn::IfThenElse(lt_i, top_idx[q][k], displaced_idx);
          top_v[q][k] = new_k;
          top_idx[q][k] = new_k_idx;
        }
      };

      size_t p = 0;
      for (; p + 4 <= num_panels; p += 4) {
        const size_t base0 = p * kPoints;
        const size_t base1 = base0 + kPoints;
        const size_t base2 = base1 + kPoints;
        const size_t base3 = base2 + kPoints;

        hn::Vec<Int32> acc0[kMqK], acc1[kMqK], acc2[kMqK], acc3[kMqK];
        if (use_hamming) {
          HammingMicroKernelAccumulate4<kMqK>(
              u8, i8, i32, hamming_qbuf_k, qbuf_tile_stride_k,
              leaders.panel_data() + p * panel_bytes,
              leaders.panel_data() + (p + 1) * panel_bytes,
              leaders.panel_data() + (p + 2) * panel_bytes,
              leaders.panel_data() + (p + 3) * panel_bytes,
              num_hamming_tiles, N, acc0, acc1, acc2, acc3);
        } else {
          MicroKernelAccumulate4<kMqK>(
              u8, i8, i32, query_ptrs,
              leaders.panel_data() + p * panel_bytes,
              leaders.panel_data() + (p + 1) * panel_bytes,
              leaders.panel_data() + (p + 2) * panel_bytes,
              leaders.panel_data() + (p + 3) * panel_bytes, total_tiles, N, acc0,
              acc1, acc2, acc3);
        }

        const auto norms0 = hn::Neg(hn::Load(f32, leaders.norms() + base0));
        const auto norms1 = hn::Neg(hn::Load(f32, leaders.norms() + base1));
        const auto norms2 = hn::Neg(hn::Load(f32, leaders.norms() + base2));
        const auto norms3 = hn::Neg(hn::Load(f32, leaders.norms() + base3));
        for (size_t q = 0; q < actual_mq; ++q) {
          const auto d0 = hn::Mul(acc_to_dot(acc0[q], q), norms0);
          const auto d1 = hn::Mul(acc_to_dot(acc1[q], q), norms1);
          const auto d2 = hn::Mul(acc_to_dot(acc2[q], q), norms2);
          const auto d3 = hn::Mul(acc_to_dot(acc3[q], q), norms3);
          update_topk(q, d0, base0);
          update_topk(q, d1, base1);
          update_topk(q, d2, base2);
          update_topk(q, d3, base3);
        }
      }

      for (; p + 2 <= num_panels; p += 2) {
        const size_t base0 = p * kPoints;
        const size_t base1 = base0 + kPoints;

        hn::Vec<Int32> acc0[kMqK], acc1[kMqK];
        if (use_hamming) {
          HammingMicroKernelAccumulate<kMqK>(
              u8, i8, i32, hamming_qbuf_k, qbuf_tile_stride_k,
              leaders.panel_data() + p * panel_bytes,
              leaders.panel_data() + (p + 1) * panel_bytes,
              num_hamming_tiles, N, acc0, acc1);
        } else {
          MicroKernelAccumulate<kMqK>(
              u8, i8, i32, query_ptrs,
              leaders.panel_data() + p * panel_bytes,
              leaders.panel_data() + (p + 1) * panel_bytes, total_tiles, N, acc0,
              acc1);
        }

        const auto norms0 = hn::Neg(hn::Load(f32, leaders.norms() + base0));
        const auto norms1 = hn::Neg(hn::Load(f32, leaders.norms() + base1));
        for (size_t q = 0; q < actual_mq; ++q) {
          const auto d0 = hn::Mul(acc_to_dot(acc0[q], q), norms0);
          const auto d1 = hn::Mul(acc_to_dot(acc1[q], q), norms1);
          update_topk(q, d0, base0);
          update_topk(q, d1, base1);
        }
      }

      for (; p < num_panels; ++p) {
        const size_t base = p * kPoints;

        hn::Vec<Int32> acc[kMqK];
        if (use_hamming) {
          HammingMicroKernelAccumulate1<kMqK>(
              u8, i8, i32, hamming_qbuf_k, qbuf_tile_stride_k,
              leaders.panel_data() + p * panel_bytes,
              num_hamming_tiles, N, acc);
        } else {
          MicroKernelAccumulate1<kMqK>(u8, i8, i32, query_ptrs,
                                       leaders.panel_data() + p * panel_bytes,
                                       total_tiles, N, acc);
        }

        const auto norms_v = hn::Neg(hn::Load(f32, leaders.norms() + base));
        for (size_t q = 0; q < actual_mq; ++q) {
          const auto d = hn::Mul(acc_to_dot(acc[q], q), norms_v);
          update_topk(q, d, base);
        }
      }

      for (size_t q = 0; q < actual_mq; ++q) {
        HWY_ALIGN float all_scores[K * HWY_MAX_BYTES / sizeof(float)];
        HWY_ALIGN int32_t all_indices[K * HWY_MAX_BYTES / sizeof(int32_t)];
        for (size_t k = 0; k < K; ++k) {
          hn::Store(top_v[q][k], f32, all_scores + k * NF);
          hn::Store(top_idx[q][k], i32, all_indices + k * NF);
        }
        auto& result = topk_results[q_start + q];
        for (size_t j = 0; j < K * NF; ++j) {
          if (all_scores[j] < inf) {
            InsertTopK(result, all_scores[j], all_indices[j]);
          }
        }
        {
          const float qn = workspace->q_norms[q_start + q];
          for (size_t k = 0; k < K; ++k) {
            result.scores[k] *= qn;
          }
        }
      }
    }

  } else {
    DecodeGatheredQueries(quantized_points, query_indices, workspace);

    const size_t min_size = K * num_queries * NF;
    workspace->min_scores.resize(min_size);
    workspace->min_indices.resize(min_size);
    std::fill(workspace->min_scores.begin(), workspace->min_scores.end(), inf);
    std::fill(workspace->min_indices.begin(), workspace->min_indices.end(),
              int32_t{-1});

    for (size_t p_block = 0; p_block < num_panels;
         p_block += panels_per_block) {
      const size_t p_end = std::min(p_block + panels_per_block, num_panels);

      for (size_t q_start = 0; q_start < num_queries; q_start += kMqK) {
        const size_t actual_mq = std::min(kMqK, num_queries - q_start);

        const int8_t* query_ptrs[kMqK];
        const uint8_t* hamming_query_ptrs[kMqK];
        float q_norms[kMqK];
        for (size_t q = 0; q < actual_mq; ++q) {
          if (use_hamming) {
            hamming_query_ptrs[q] =
                raw + query_indices[q_start + q] * stride;
          } else {
            query_ptrs[q] =
                workspace->all_q_decoded.data() + (q_start + q) * decoded_dim;
          }
          q_norms[q] = workspace->q_norms[q_start + q];
        }
        for (size_t q = actual_mq; q < kMqK; ++q) {
          if (use_hamming) {
            hamming_query_ptrs[q] = hamming_query_ptrs[actual_mq - 1];
          } else {
            query_ptrs[q] = query_ptrs[actual_mq - 1];
          }
          q_norms[q] = 0.0f;
        }

        const size_t qbuf_tile_stride_k2 = num_hamming_tiles * N;
        HWY_ALIGN uint8_t hamming_qbuf_k2[kMqK * 64 * 64];
        if (use_hamming) {
          PreBroadcastHammingQueries<kMqK>(u8, i32, hamming_query_ptrs,
              num_hamming_tiles, bytes_per_point, N, hamming_qbuf_k2);
        }

        hn::Vec<Float> neg_corr[kMqK];
        if (!use_hamming) {
          for (size_t q = 0; q < actual_mq; ++q) {
            neg_corr[q] =
                hn::Set(f32, 128.0f * workspace->all_q_byte_sums[q_start + q]);
          }
        }
        for (size_t q = use_hamming ? 0 : actual_mq; q < kMqK; ++q) {
          neg_corr[q] = hn::Zero(f32);
        }

        auto acc_to_dot = [&](hn::Vec<Int32>& a, size_t q) -> hn::Vec<Float> {
          if (use_hamming) {
            const auto h_f = hn::ConvertTo(f32, a);
            return hn::MulAdd(hn::Set(f32, -2.0f * 16129.0f), h_f,
                              hn::Set(f32, 16129.0f * static_cast<float>(dims)));
          } else {
            return hn::Sub(hn::ConvertTo(f32, a), neg_corr[q]);
          }
        };

        const auto inf_v = hn::Set(f32, inf);
        hn::Vec<Float> top_v[kMqK][K];
        hn::Vec<Int32> top_idx[kMqK][K];
        for (size_t q = 0; q < actual_mq; ++q) {
          for (size_t k = 0; k < K; ++k) {
            const size_t off = ((q_start + q) * K + k) * NF;
            top_v[q][k] = hn::Load(f32, workspace->min_scores.data() + off);
            top_idx[q][k] = hn::Load(i32, workspace->min_indices.data() + off);
          }
        }
        for (size_t q = actual_mq; q < kMqK; ++q) {
          for (size_t k = 0; k < K; ++k) {
            top_v[q][k] = inf_v;
            top_idx[q][k] = hn::Set(i32, -1);
          }
        }

        const size_t num_db_points = leaders.num_points();
        const auto update_topk = [&](size_t q, hn::Vec<Float> s,
                                     size_t panel_base) HWY_ATTR {
          const auto base_v =
              hn::Add(hn::Set(i32, static_cast<int32_t>(panel_base)), lane_idx);
          if (HWY_UNLIKELY(panel_base + kPoints > num_db_points)) {
            const auto valid = hn::RebindMask(
                f32, hn::Lt(base_v,
                            hn::Set(i32, static_cast<int32_t>(num_db_points))));
            s = hn::IfThenElse(valid, s, hn::Set(f32, inf));
          }
          auto displaced = s;
          auto displaced_idx = base_v;
          for (size_t k = 0; k < K; ++k) {
            const auto lt = hn::Lt(displaced, top_v[q][k]);
            const auto lt_i = hn::RebindMask(i32, lt);
            const auto new_k = hn::IfThenElse(lt, displaced, top_v[q][k]);
            const auto new_k_idx =
                hn::IfThenElse(lt_i, displaced_idx, top_idx[q][k]);
            displaced = hn::IfThenElse(lt, top_v[q][k], displaced);
            displaced_idx = hn::IfThenElse(lt_i, top_idx[q][k], displaced_idx);
            top_v[q][k] = new_k;
            top_idx[q][k] = new_k_idx;
          }
        };

        size_t p = p_block;
        for (; p + 4 <= p_end; p += 4) {
          const size_t base0 = p * kPoints;
          const size_t base1 = base0 + kPoints;
          const size_t base2 = base1 + kPoints;
          const size_t base3 = base2 + kPoints;

          hn::Vec<Int32> acc0[kMqK], acc1[kMqK], acc2[kMqK], acc3[kMqK];
          if (use_hamming) {
            HammingMicroKernelAccumulate4<kMqK>(
                u8, i8, i32, hamming_qbuf_k2, qbuf_tile_stride_k2,
                leaders.panel_data() + p * panel_bytes,
                leaders.panel_data() + (p + 1) * panel_bytes,
                leaders.panel_data() + (p + 2) * panel_bytes,
                leaders.panel_data() + (p + 3) * panel_bytes,
                num_hamming_tiles, N, acc0, acc1, acc2, acc3);
          } else {
            MicroKernelAccumulate4<kMqK>(
                u8, i8, i32, query_ptrs,
                leaders.panel_data() + p * panel_bytes,
                leaders.panel_data() + (p + 1) * panel_bytes,
                leaders.panel_data() + (p + 2) * panel_bytes,
                leaders.panel_data() + (p + 3) * panel_bytes, total_tiles, N,
                acc0, acc1, acc2, acc3);
          }

          const auto norms0 = hn::Neg(hn::Load(f32, leaders.norms() + base0));
          const auto norms1 = hn::Neg(hn::Load(f32, leaders.norms() + base1));
          const auto norms2 = hn::Neg(hn::Load(f32, leaders.norms() + base2));
          const auto norms3 = hn::Neg(hn::Load(f32, leaders.norms() + base3));
          for (size_t q = 0; q < actual_mq; ++q) {
            const auto d0 = hn::Mul(acc_to_dot(acc0[q], q), norms0);
            const auto d1 = hn::Mul(acc_to_dot(acc1[q], q), norms1);
            const auto d2 = hn::Mul(acc_to_dot(acc2[q], q), norms2);
            const auto d3 = hn::Mul(acc_to_dot(acc3[q], q), norms3);
            update_topk(q, d0, base0);
            update_topk(q, d1, base1);
            update_topk(q, d2, base2);
            update_topk(q, d3, base3);
          }
        }

        for (; p + 2 <= p_end; p += 2) {
          const size_t base0 = p * kPoints;
          const size_t base1 = base0 + kPoints;

          hn::Vec<Int32> acc0[kMqK], acc1[kMqK];
          if (use_hamming) {
            HammingMicroKernelAccumulate<kMqK>(
                u8, i8, i32, hamming_qbuf_k2, qbuf_tile_stride_k2,
                leaders.panel_data() + p * panel_bytes,
                leaders.panel_data() + (p + 1) * panel_bytes,
                num_hamming_tiles, N, acc0, acc1);
          } else {
            MicroKernelAccumulate<kMqK>(
                u8, i8, i32, query_ptrs,
                leaders.panel_data() + p * panel_bytes,
                leaders.panel_data() + (p + 1) * panel_bytes, total_tiles, N,
                acc0, acc1);
          }

          const auto norms0 = hn::Neg(hn::Load(f32, leaders.norms() + base0));
          const auto norms1 = hn::Neg(hn::Load(f32, leaders.norms() + base1));
          for (size_t q = 0; q < actual_mq; ++q) {
            const auto d0 = hn::Mul(acc_to_dot(acc0[q], q), norms0);
            const auto d1 = hn::Mul(acc_to_dot(acc1[q], q), norms1);
            update_topk(q, d0, base0);
            update_topk(q, d1, base1);
          }
        }

        for (; p < p_end; ++p) {
          const size_t base = p * kPoints;

          hn::Vec<Int32> acc[kMqK];
          if (use_hamming) {
            HammingMicroKernelAccumulate1<kMqK>(
                u8, i8, i32, hamming_qbuf_k2, qbuf_tile_stride_k2,
                leaders.panel_data() + p * panel_bytes,
                num_hamming_tiles, N, acc);
          } else {
            MicroKernelAccumulate1<kMqK>(u8, i8, i32, query_ptrs,
                                         leaders.panel_data() + p * panel_bytes,
                                         total_tiles, N, acc);
          }

          const auto norms_v = hn::Neg(hn::Load(f32, leaders.norms() + base));
          for (size_t q = 0; q < actual_mq; ++q) {
            const auto d = hn::Mul(acc_to_dot(acc[q], q), norms_v);
            update_topk(q, d, base);
          }
        }

        for (size_t q = 0; q < actual_mq; ++q) {
          for (size_t k = 0; k < K; ++k) {
            const size_t off = ((q_start + q) * K + k) * NF;
            hn::Store(top_v[q][k], f32, workspace->min_scores.data() + off);
            hn::Store(top_idx[q][k], i32, workspace->min_indices.data() + off);
          }
        }
      }
    }

    for (size_t qi = 0; qi < num_queries; ++qi) {
      HWY_ALIGN float all_scores[K * HWY_MAX_BYTES / sizeof(float)];
      HWY_ALIGN int32_t all_indices[K * HWY_MAX_BYTES / sizeof(int32_t)];
      for (size_t k = 0; k < K; ++k) {
        const size_t off = (qi * K + k) * NF;
        hn::Store(hn::Load(f32, workspace->min_scores.data() + off), f32,
                  all_scores + k * NF);
        hn::Store(hn::Load(i32, workspace->min_indices.data() + off), i32,
                  all_indices + k * NF);
      }
      auto& result = topk_results[qi];
      for (size_t j = 0; j < K * NF; ++j) {
        if (all_scores[j] < inf) {
          InsertTopK(result, all_scores[j], all_indices[j]);
        }
      }
      {
        const float qn = workspace->q_norms[qi];
        for (size_t k = 0; k < K; ++k) {
          result.scores[k] *= qn;
        }
      }
    }
  }

}

template <size_t K>
void AssignToLeadersConsecutiveTopKImpl(
    const RsqBasePoints& quantized_points, size_t query_begin,
    size_t query_end, const PreTransposedRsqDataset& leaders,
    std::vector<TopKResult<K>>& topk_results,
    AssignToLeadersWorkspace* workspace) {
  static constexpr size_t kMqK = 6;

  using Int8 = hn::ScalableTag<int8_t>;
  using Uint8 = hn::ScalableTag<uint8_t>;
  using Int32 = hn::ScalableTag<int32_t>;
  using Float = hn::ScalableTag<float>;

  const Int8 i8;
  const Uint8 u8;
  const Int32 i32;
  const Float f32;

  const bool is_8bit = quantized_points.is_8bit();
  const bool is_1bit = quantized_points.is_1bit();
  const bool use_hamming = is_1bit;
  const size_t N = hn::Lanes(i8);
  const size_t kPoints = hn::Lanes(i32);
  const size_t num_queries = query_end - query_begin;
  const size_t bytes_per_point = quantized_points.num_bytes_per_datapoint();
  const size_t dims = quantized_points.dimensionality();
  const size_t stride = quantized_points.stride();
  const size_t num_dim_chunks = hwy::DivCeil(bytes_per_point, N);
  const size_t num_hamming_tiles = hwy::DivCeil(bytes_per_point, 4);
  const size_t total_tiles = use_hamming ? num_hamming_tiles
                           : is_8bit ? num_dim_chunks * (N / 4)
                                     : num_dim_chunks * (2 * N);
  const size_t decoded_dim = is_8bit ? num_dim_chunks * N
                                     : num_dim_chunks * 8 * N;

  const uint8_t* raw = quantized_points.quantized_data();

  if (!use_hamming) {
    DecodeConsecutiveQueries(quantized_points, query_begin, query_end,
                             workspace);
  } else {
    workspace->q_norms.resize(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
      workspace->q_norms[i] =
          quantized_points.norm_scaling_factor(query_begin + i);
    }
  }

  const size_t num_panels = leaders.num_panels();
  const size_t panel_bytes = leaders.panel_bytes();

  topk_results.assign(num_queries, TopKResult<K>{});

  static constexpr size_t kL2Budget = 768 * 1024;
  const size_t panels_per_block =
      std::max(size_t{4}, (kL2Budget / panel_bytes) & ~size_t{3});

  for (size_t p_block = 0; p_block < num_panels; p_block += panels_per_block) {
    const size_t p_end = std::min(p_block + panels_per_block, num_panels);

    for (size_t q_start = 0; q_start < num_queries; q_start += kMqK) {
      const size_t actual_mq = std::min(kMqK, num_queries - q_start);

      const int8_t* query_ptrs[kMqK];
      const uint8_t* hamming_query_ptrs[kMqK];
      float q_norms[kMqK];
      for (size_t q = 0; q < actual_mq; ++q) {
        if (use_hamming) {
          hamming_query_ptrs[q] = raw + (query_begin + q_start + q) * stride;
        } else {
          query_ptrs[q] =
              workspace->all_q_decoded.data() + (q_start + q) * decoded_dim;
        }
        q_norms[q] = workspace->q_norms[q_start + q];
      }
      for (size_t q = actual_mq; q < kMqK; ++q) {
        if (use_hamming) {
          hamming_query_ptrs[q] = hamming_query_ptrs[actual_mq - 1];
        } else {
          query_ptrs[q] = query_ptrs[actual_mq - 1];
        }
        q_norms[q] = 0.0f;
      }

      const size_t qbuf_tile_stride_c = num_hamming_tiles * N;
      HWY_ALIGN uint8_t hamming_qbuf_c[kMqK * 64 * 64];
      if (use_hamming) {
        PreBroadcastHammingQueries<kMqK>(u8, i32, hamming_query_ptrs,
            num_hamming_tiles, bytes_per_point, N, hamming_qbuf_c);
      }

      hn::Vec<Float> neg_corr[kMqK];
      if (!use_hamming) {
        for (size_t q = 0; q < actual_mq; ++q) {
          neg_corr[q] =
              hn::Set(f32, 128.0f * workspace->all_q_byte_sums[q_start + q]);
        }
      }
      for (size_t q = use_hamming ? 0 : actual_mq; q < kMqK; ++q) {
        neg_corr[q] = hn::Zero(f32);
      }

      auto acc_to_dot = [&](hn::Vec<Int32>& a, size_t q) -> hn::Vec<Float> {
        if (use_hamming) {
          const auto h_f = hn::ConvertTo(f32, a);
          return hn::MulAdd(hn::Set(f32, -2.0f * 16129.0f), h_f,
                            hn::Set(f32, 16129.0f * static_cast<float>(dims)));
        } else {
          return hn::Sub(hn::ConvertTo(f32, a), neg_corr[q]);
        }
      };

      const size_t num_db_points = leaders.num_points();
      const auto lane_idx = hn::Iota(i32, 0);
      const auto mask_padding = [&](hn::Vec<Float> s, size_t base)
                                    HWY_ATTR -> hn::Vec<Float> {
        if (HWY_LIKELY(base + kPoints <= num_db_points)) return s;
        const auto valid = hn::RebindMask(
            f32,
            hn::Lt(hn::Add(hn::Set(i32, static_cast<int32_t>(base)), lane_idx),
                   hn::Set(i32, static_cast<int32_t>(num_db_points))));
        return hn::IfThenElse(valid, s,
                              hn::Set(f32, std::numeric_limits<float>::max()));
      };

      const auto score_epilogue = [&](size_t q, hn::Vec<Float> s,
                                      size_t base) HWY_ATTR {
        s = mask_padding(s, base);
        auto& result = topk_results[q_start + q];
        const auto thresh = hn::Set(f32, result.scores[K - 1]);
        const auto lt_mask = hn::Lt(s, thresh);
        if (HWY_LIKELY(hn::AllFalse(f32, lt_mask))) return;
        const size_t count = hn::CountTrue(f32, lt_mask);
        HWY_ALIGN float compressed_scores[HWY_MAX_BYTES / sizeof(float)];
        HWY_ALIGN int32_t compressed_indices[HWY_MAX_BYTES / sizeof(int32_t)];
        hn::CompressStore(s, lt_mask, f32, compressed_scores);
        const auto base_idx =
            hn::Add(hn::Set(i32, static_cast<int32_t>(base)), hn::Iota(i32, 0));
        hn::CompressStore(base_idx, hn::RebindMask(i32, lt_mask), i32,
                          compressed_indices);
        for (size_t j = 0; j < count; ++j) {
          InsertTopK(result, compressed_scores[j], compressed_indices[j]);
        }
      };

      size_t p = p_block;

      for (; p + 8 <= p_end; p += 8) {
        const size_t base0 = p * kPoints;
        const size_t base1 = base0 + kPoints;
        const size_t base2 = base1 + kPoints;
        const size_t base3 = base2 + kPoints;
        const size_t base4 = base3 + kPoints;
        const size_t base5 = base4 + kPoints;
        const size_t base6 = base5 + kPoints;
        const size_t base7 = base6 + kPoints;

        hn::Vec<Int32> acc0[kMqK], acc1[kMqK], acc2[kMqK], acc3[kMqK];
        hn::Vec<Int32> acc4[kMqK], acc5[kMqK], acc6[kMqK], acc7[kMqK];
        if (use_hamming) {
          HammingMicroKernelAccumulate4<kMqK>(
              u8, i8, i32, hamming_qbuf_c, qbuf_tile_stride_c,
              leaders.panel_data() + p * panel_bytes,
              leaders.panel_data() + (p + 1) * panel_bytes,
              leaders.panel_data() + (p + 2) * panel_bytes,
              leaders.panel_data() + (p + 3) * panel_bytes,
              num_hamming_tiles, N, acc0, acc1, acc2, acc3);
          HammingMicroKernelAccumulate4<kMqK>(
              u8, i8, i32, hamming_qbuf_c, qbuf_tile_stride_c,
              leaders.panel_data() + (p + 4) * panel_bytes,
              leaders.panel_data() + (p + 5) * panel_bytes,
              leaders.panel_data() + (p + 6) * panel_bytes,
              leaders.panel_data() + (p + 7) * panel_bytes,
              num_hamming_tiles, N, acc4, acc5, acc6, acc7);
        } else {
          MicroKernelAccumulate4<kMqK>(
              u8, i8, i32, query_ptrs,
              leaders.panel_data() + p * panel_bytes,
              leaders.panel_data() + (p + 1) * panel_bytes,
              leaders.panel_data() + (p + 2) * panel_bytes,
              leaders.panel_data() + (p + 3) * panel_bytes, total_tiles, N,
              acc0, acc1, acc2, acc3);
          MicroKernelAccumulate4<kMqK>(
              u8, i8, i32, query_ptrs,
              leaders.panel_data() + (p + 4) * panel_bytes,
              leaders.panel_data() + (p + 5) * panel_bytes,
              leaders.panel_data() + (p + 6) * panel_bytes,
              leaders.panel_data() + (p + 7) * panel_bytes, total_tiles, N,
              acc4, acc5, acc6, acc7);
        }

        const auto norms0 = hn::Neg(hn::Load(f32, leaders.norms() + base0));
        const auto norms1 = hn::Neg(hn::Load(f32, leaders.norms() + base1));
        const auto norms2 = hn::Neg(hn::Load(f32, leaders.norms() + base2));
        const auto norms3 = hn::Neg(hn::Load(f32, leaders.norms() + base3));
        const auto norms4 = hn::Neg(hn::Load(f32, leaders.norms() + base4));
        const auto norms5 = hn::Neg(hn::Load(f32, leaders.norms() + base5));
        const auto norms6 = hn::Neg(hn::Load(f32, leaders.norms() + base6));
        const auto norms7 = hn::Neg(hn::Load(f32, leaders.norms() + base7));
        for (size_t q = 0; q < actual_mq; ++q) {
          const auto s0 = hn::Mul(acc_to_dot(acc0[q], q), norms0);
          const auto s1 = hn::Mul(acc_to_dot(acc1[q], q), norms1);
          const auto s2 = hn::Mul(acc_to_dot(acc2[q], q), norms2);
          const auto s3 = hn::Mul(acc_to_dot(acc3[q], q), norms3);
          const auto s4 = hn::Mul(acc_to_dot(acc4[q], q), norms4);
          const auto s5 = hn::Mul(acc_to_dot(acc5[q], q), norms5);
          const auto s6 = hn::Mul(acc_to_dot(acc6[q], q), norms6);
          const auto s7 = hn::Mul(acc_to_dot(acc7[q], q), norms7);

          const auto cs0 = s0;
          const auto cs1 = s1;
          const auto cs2 = s2;
          const auto cs3 = s3;
          const auto cs4 = s4;
          const auto cs5 = s5;
          const auto cs6 = s6;
          const auto cs7 = s7;
          const auto smin = hn::Min(hn::Min(hn::Min(cs0, cs1), hn::Min(cs2, cs3)),
                                    hn::Min(hn::Min(cs4, cs5), hn::Min(cs6, cs7)));
          const auto thresh =
              hn::Set(f32, topk_results[q_start + q].scores[K - 1]);
          if (HWY_LIKELY(hn::AllFalse(f32, hn::Lt(smin, thresh)))) continue;
          score_epilogue(q, cs0, base0);
          score_epilogue(q, cs1, base1);
          score_epilogue(q, cs2, base2);
          score_epilogue(q, cs3, base3);
          score_epilogue(q, cs4, base4);
          score_epilogue(q, cs5, base5);
          score_epilogue(q, cs6, base6);
          score_epilogue(q, cs7, base7);
        }
      }

      for (; p + 4 <= p_end; p += 4) {
        const size_t base0 = p * kPoints;
        const size_t base1 = base0 + kPoints;
        const size_t base2 = base1 + kPoints;
        const size_t base3 = base2 + kPoints;

        hn::Vec<Int32> acc0[kMqK], acc1[kMqK], acc2[kMqK], acc3[kMqK];
        if (use_hamming) {
          HammingMicroKernelAccumulate4<kMqK>(
              u8, i8, i32, hamming_qbuf_c, qbuf_tile_stride_c,
              leaders.panel_data() + p * panel_bytes,
              leaders.panel_data() + (p + 1) * panel_bytes,
              leaders.panel_data() + (p + 2) * panel_bytes,
              leaders.panel_data() + (p + 3) * panel_bytes,
              num_hamming_tiles, N, acc0, acc1, acc2, acc3);
        } else {
          MicroKernelAccumulate4<kMqK>(
              u8, i8, i32, query_ptrs,
              leaders.panel_data() + p * panel_bytes,
              leaders.panel_data() + (p + 1) * panel_bytes,
              leaders.panel_data() + (p + 2) * panel_bytes,
              leaders.panel_data() + (p + 3) * panel_bytes, total_tiles, N,
              acc0, acc1, acc2, acc3);
        }

        const auto norms0 = hn::Neg(hn::Load(f32, leaders.norms() + base0));
        const auto norms1 = hn::Neg(hn::Load(f32, leaders.norms() + base1));
        const auto norms2 = hn::Neg(hn::Load(f32, leaders.norms() + base2));
        const auto norms3 = hn::Neg(hn::Load(f32, leaders.norms() + base3));
        for (size_t q = 0; q < actual_mq; ++q) {
          const auto d0 = hn::Mul(acc_to_dot(acc0[q], q), norms0);
          const auto d1 = hn::Mul(acc_to_dot(acc1[q], q), norms1);
          const auto d2 = hn::Mul(acc_to_dot(acc2[q], q), norms2);
          const auto d3 = hn::Mul(acc_to_dot(acc3[q], q), norms3);
          score_epilogue(q, d0, base0);
          score_epilogue(q, d1, base1);
          score_epilogue(q, d2, base2);
          score_epilogue(q, d3, base3);
        }
      }

      for (; p + 2 <= p_end; p += 2) {
        const size_t base0 = p * kPoints;
        const size_t base1 = base0 + kPoints;

        hn::Vec<Int32> acc0[kMqK], acc1[kMqK];
        if (use_hamming) {
          HammingMicroKernelAccumulate<kMqK>(
              u8, i8, i32, hamming_qbuf_c, qbuf_tile_stride_c,
              leaders.panel_data() + p * panel_bytes,
              leaders.panel_data() + (p + 1) * panel_bytes,
              num_hamming_tiles, N, acc0, acc1);
        } else {
          MicroKernelAccumulate<kMqK>(
              u8, i8, i32, query_ptrs,
              leaders.panel_data() + p * panel_bytes,
              leaders.panel_data() + (p + 1) * panel_bytes, total_tiles, N,
              acc0, acc1);
        }

        const auto norms0 = hn::Neg(hn::Load(f32, leaders.norms() + base0));
        const auto norms1 = hn::Neg(hn::Load(f32, leaders.norms() + base1));
        for (size_t q = 0; q < actual_mq; ++q) {
          const auto d0 = hn::Mul(acc_to_dot(acc0[q], q), norms0);
          const auto d1 = hn::Mul(acc_to_dot(acc1[q], q), norms1);
          score_epilogue(q, d0, base0);
          score_epilogue(q, d1, base1);
        }
      }

      for (; p < p_end; ++p) {
        const size_t base = p * kPoints;

        hn::Vec<Int32> acc[kMqK];
        if (use_hamming) {
          HammingMicroKernelAccumulate1<kMqK>(
              u8, i8, i32, hamming_qbuf_c, qbuf_tile_stride_c,
              leaders.panel_data() + p * panel_bytes,
              num_hamming_tiles, N, acc);
        } else {
          MicroKernelAccumulate1<kMqK>(u8, i8, i32, query_ptrs,
                                       leaders.panel_data() + p * panel_bytes,
                                       total_tiles, N, acc);
        }

        const auto norms_v = hn::Neg(hn::Load(f32, leaders.norms() + base));
        for (size_t q = 0; q < actual_mq; ++q) {
          const auto d = hn::Mul(acc_to_dot(acc[q], q), norms_v);
          score_epilogue(q, d, base);
        }
      }
    }
  }

  {
    for (size_t qi = 0; qi < num_queries; ++qi) {
      const float qn = workspace->q_norms[qi];
      for (size_t k = 0; k < K; ++k) {
        topk_results[qi].scores[k] *= qn;
      }
    }
  }

}

void AssignToLeadersTopK3Impl(const RsqBasePoints& qp,
                              absl::Span<const int> qi,
                              const PreTransposedRsqDataset& l,
                              std::vector<TopKResult<3>>& r,
                              AssignToLeadersWorkspace* w) {
  AssignToLeadersTopKImpl<3>(qp, qi, l, r, w);
}
void AssignToLeadersTopK10Impl(const RsqBasePoints& qp,
                               absl::Span<const int> qi,
                               const PreTransposedRsqDataset& l,
                               std::vector<TopKResult<10>>& r,
                               AssignToLeadersWorkspace* w) {
  AssignToLeadersTopKImpl<10>(qp, qi, l, r, w);
}
void AssignToLeadersConsecutiveTopK3Impl(
    const RsqBasePoints& qp, size_t qb, size_t qe,
    const PreTransposedRsqDataset& l,
    std::vector<TopKResult<3>>& r, AssignToLeadersWorkspace* w) {
  AssignToLeadersConsecutiveTopKImpl<3>(qp, qb, qe, l, r, w);
}
void AssignToLeadersConsecutiveTopK10Impl(
    const RsqBasePoints& qp, size_t qb, size_t qe,
    const PreTransposedRsqDataset& l,
    std::vector<TopKResult<10>>& r, AssignToLeadersWorkspace* w) {
  AssignToLeadersConsecutiveTopKImpl<10>(qp, qb, qe, l, r, w);
}

}
}

}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace pipnn {
HWY_EXPORT(AssignToLeadersTop1Impl);
HWY_EXPORT(AssignToLeadersTopK3Impl);
HWY_EXPORT(AssignToLeadersTopK10Impl);
HWY_EXPORT(AssignToLeadersConsecutiveTopK3Impl);
HWY_EXPORT(AssignToLeadersConsecutiveTopK10Impl);

void AssignToLeadersTop1(const RsqBasePoints& quantized_points,
                         absl::Span<const int> query_indices,
                         const PreTransposedRsqDataset& leaders,
                         std::vector<TopKResult<1>>& top1_results,
                         AssignToLeadersWorkspace* workspace) {
  AssignToLeadersWorkspace local_ws;
  if (!workspace) workspace = &local_ws;
  HWY_DYNAMIC_DISPATCH(AssignToLeadersTop1Impl)
  (quantized_points, query_indices, leaders, top1_results, workspace);
}

template <>
void AssignToLeadersTopK<3>(
    const RsqBasePoints& quantized_points,
    absl::Span<const int> query_indices,
    const PreTransposedRsqDataset& leaders,
    std::vector<TopKResult<3>>& topk_results,
    AssignToLeadersWorkspace* workspace) {
  AssignToLeadersWorkspace local_ws;
  if (!workspace) workspace = &local_ws;
  HWY_DYNAMIC_DISPATCH(AssignToLeadersTopK3Impl)
  (quantized_points, query_indices, leaders, topk_results, workspace);
}

template <>
void AssignToLeadersTopK<10>(
    const RsqBasePoints& quantized_points,
    absl::Span<const int> query_indices,
    const PreTransposedRsqDataset& leaders,
    std::vector<TopKResult<10>>& topk_results,
    AssignToLeadersWorkspace* workspace) {
  AssignToLeadersWorkspace local_ws;
  if (!workspace) workspace = &local_ws;
  HWY_DYNAMIC_DISPATCH(AssignToLeadersTopK10Impl)
  (quantized_points, query_indices, leaders, topk_results, workspace);
}

template <>
void AssignToLeadersConsecutiveTopK<3>(
    const RsqBasePoints& quantized_points, size_t query_begin,
    size_t query_end, const PreTransposedRsqDataset& leaders,
    std::vector<TopKResult<3>>& topk_results,
    AssignToLeadersWorkspace* workspace) {
  AssignToLeadersWorkspace local_ws;
  if (!workspace) workspace = &local_ws;
  HWY_DYNAMIC_DISPATCH(AssignToLeadersConsecutiveTopK3Impl)
  (quantized_points, query_begin, query_end, leaders, topk_results, workspace);
}

template <>
void AssignToLeadersConsecutiveTopK<10>(
    const RsqBasePoints& quantized_points, size_t query_begin,
    size_t query_end, const PreTransposedRsqDataset& leaders,
    std::vector<TopKResult<10>>& topk_results,
    AssignToLeadersWorkspace* workspace) {
  AssignToLeadersWorkspace local_ws;
  if (!workspace) workspace = &local_ws;
  HWY_DYNAMIC_DISPATCH(AssignToLeadersConsecutiveTopK10Impl)
  (quantized_points, query_begin, query_end, leaders, topk_results, workspace);
}

}

#endif

