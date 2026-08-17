#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "rsq_encoder.h"
#include "absl/base/prefetch.h"
#include "pipnn_rsq.h"
#include "absl/types/span.h"
#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "processleaf_rsq.cc"

#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace pipnn {
namespace HWY_NAMESPACE {
namespace {
#include "pipnn_rsq-inl.h"

HWY_INLINE void ScalarInsertTopK(float* HWY_RESTRICT s, int32_t* HWY_RESTRICT iv,
                                 size_t K, float val, int32_t idx) {
  if (val >= s[K - 1]) return;
  size_t pos = K - 1;
  while (pos > 0 && val < s[pos - 1]) {
    s[pos] = s[pos - 1];
    iv[pos] = iv[pos - 1];
    --pos;
  }
  s[pos] = val;
  iv[pos] = idx;
}

template <size_t K>
HWY_INLINE void InsertTopKFused(
    const hn::ScalableTag<float> f32, const hn::ScalableTag<int32_t> i32,
    hn::Vec<hn::ScalableTag<float>>* HWY_RESTRICT m,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT iv,
    const hn::Vec<hn::ScalableTag<float>>& sv,
    const hn::Vec<hn::ScalableTag<int32_t>>& newidx) {
  using Float = hn::ScalableTag<float>;
  hn::Mask<Float> lt[K];
  hn::Mask<Float> cum[K];
  lt[0] = hn::Lt(sv, m[0]);
  cum[0] = lt[0];
  for (size_t j = 1; j < K; ++j) {
    lt[j] = hn::AndNot(cum[j - 1], hn::Lt(sv, m[j]));
    cum[j] = hn::Or(cum[j - 1], lt[j]);
  }

  for (size_t j = K - 1; j >= 1; --j) {
    m[j] = hn::IfThenElse(cum[j - 1], m[j - 1],
                          hn::IfThenElse(lt[j], sv, m[j]));
    iv[j] = hn::IfThenElse(hn::RebindMask(i32, cum[j - 1]), iv[j - 1],
                           hn::IfThenElse(hn::RebindMask(i32, lt[j]), newidx,
                                          iv[j]));
  }
  m[0] = hn::IfThenElse(lt[0], sv, m[0]);
  iv[0] = hn::IfThenElse(hn::RebindMask(i32, lt[0]), newidx, iv[0]);
}

template <size_t K, bool kIsHamming = false>
HWY_INLINE void ScoreAndScatterEpilogueK(
    const hn::ScalableTag<float> f32, const hn::ScalableTag<int32_t> i32,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc, size_t actual_mq,
    const hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT neg_correction,
    const hn::Vec<hn::ScalableTag<float>>& neg_db_norms_v,
    const float* HWY_RESTRICT q_norms, const size_t* HWY_RESTRICT q_abs_indices,
    size_t base, size_t kP, size_t kPoints,
    const hn::Vec<hn::ScalableTag<int32_t>>& lane_idx,
    const hn::Vec<hn::ScalableTag<float>>& inf_v,
    hn::Vec<hn::ScalableTag<float>>* HWY_RESTRICT minv,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT minidx,
    hn::Vec<hn::ScalableTag<float>>* HWY_RESTRICT dm,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT di,
    size_t dims = 0) {
  using Float = hn::ScalableTag<float>;
  const auto valid_mask_f = hn::RebindMask(
      f32, hn::Lt(lane_idx, hn::Set(i32, static_cast<int32_t>(kP))));
  const auto base_v =
      hn::Add(hn::Set(i32, static_cast<int32_t>(base)), lane_idx);
  const bool any_self = (base < q_abs_indices[actual_mq - 1] + 1) &&
                        (q_abs_indices[0] < base + kPoints);

  for (size_t q = 0; q < actual_mq; ++q) {
    hn::Vec<Float> s;
    if constexpr (kIsHamming) {
      const auto h_f = hn::ConvertTo(f32, acc[q]);
      const auto dot_f = hn::MulAdd(hn::Set(f32, -2.0f * 16129.0f), h_f,
                                    hn::Set(f32, 16129.0f * static_cast<float>(dims)));
      s = hn::Mul(dot_f, neg_db_norms_v);
    } else {
      acc[q] = hn::Add(acc[q], neg_correction[q]);
      s = hn::Mul(hn::ConvertTo(f32, acc[q]), neg_db_norms_v);
    }

    hn::Vec<Float> sq = s;
    if (kP < kPoints) sq = hn::IfThenElse(valid_mask_f, sq, inf_v);
    const size_t qi = q_abs_indices[q];
    hn::Mask<Float> self_mask_f;
    if (any_self && base <= qi && qi < base + kPoints) {
      self_mask_f = hn::RebindMask(
          f32, hn::Eq(lane_idx, hn::Set(i32, static_cast<int32_t>(qi - base))));
      sq = hn::IfThenElse(self_mask_f, inf_v, sq);
    }
    InsertTopKFused<K>(f32, i32, minv + q * K, minidx + q * K, sq, base_v);

    auto sv = hn::Mul(s, hn::Set(f32, q_norms[q]));
    if (kP < kPoints) sv = hn::IfThenElse(valid_mask_f, sv, inf_v);
    if (any_self && base <= qi && qi < base + kPoints) {
      sv = hn::IfThenElse(self_mask_f, inf_v, sv);
    }
    InsertTopKFused<K>(f32, i32, dm, di, sv,
                       hn::Set(i32, static_cast<int32_t>(qi)));
  }
}

void ProcessLeafInitImpl(const RsqBasePoints& quantized_points,
                         absl::Span<const int> indices,
                         ProcessLeafWorkspace* workspace) {
  using Int8 = hn::ScalableTag<int8_t>;
  using Uint8 = hn::ScalableTag<uint8_t>;
  using Int32 = hn::ScalableTag<int32_t>;

  const Int8 i8;
  const Uint8 u8;
  const Int32 i32;

  const bool is_8bit = quantized_points.is_8bit();
  const bool is_1bit = quantized_points.is_1bit();
  const size_t N = hn::Lanes(i8);
  const size_t kPoints = hn::Lanes(i32);
  const size_t num_points = indices.size();
  const size_t bytes_per_point = quantized_points.num_bytes_per_datapoint();
  const size_t stride = quantized_points.stride();
  const size_t num_dim_chunks = hwy::DivCeil(bytes_per_point, N);

  const size_t num_hamming_tiles = hwy::DivCeil(bytes_per_point, 4);
  const size_t tiles_per_chunk = is_8bit ? N / 4 : 2 * N;
  const size_t total_tiles = is_1bit ? num_hamming_tiles
                                     : num_dim_chunks * tiles_per_chunk;
  const size_t panel_bytes = total_tiles * N;

  const uint8_t* raw = quantized_points.quantized_data();

  workspace->all_q_byte_sums.resize(num_points);

  const size_t num_panels = hwy::DivCeil(num_points, kPoints);
  workspace->panel_data.resize(num_panels * panel_bytes);
  workspace->norms.assign(num_panels * kPoints, 0.0f);

  if (is_8bit) {
    for (size_t p = 0; p < num_panels; ++p) {
      const size_t pi_start = p * kPoints;
      const size_t kP = std::min(kPoints, num_points - pi_start);
      DecodeDbPanel8Bit(u8, raw, stride, 0, pi_start, kP,
                        kPoints, bytes_per_point, num_dim_chunks,
                        tiles_per_chunk, N,
                        workspace->panel_data.data() + p * panel_bytes,
                        indices.data());
    }
  } else if (is_1bit) {
    for (size_t p = 0; p < num_panels; ++p) {
      const size_t pi_start = p * kPoints;
      const size_t kP = std::min(kPoints, num_points - pi_start);
      BuildHammingPanel1Bit(u8, raw, stride, 0, pi_start, kP, kPoints,
                            bytes_per_point, num_hamming_tiles, N,
                            workspace->panel_data.data() + p * panel_bytes,
                            indices.data());
    }
  }
  for (size_t i = 0; i < num_points; ++i) {
    const int idx = indices[i];
    workspace->norms[i] = -quantized_points.norm_scaling_factor(idx);
    workspace->all_q_byte_sums[i] = quantized_points.byte_sum(idx);
  }
}

template <size_t K>
void ProcessLeafTopKFusedImpl(
    const RsqBasePoints& quantized_points,
    absl::Span<const int> indices,
    ProcessLeafWorkspace* workspace) {
  static constexpr size_t kMqFull = 6;

  using Int8 = hn::ScalableTag<int8_t>;
  using Uint8 = hn::ScalableTag<uint8_t>;
  using Int32 = hn::ScalableTag<int32_t>;
  const Int8 i8;
  const Uint8 u8;
  const Int32 i32;

  const bool is_8bit = quantized_points.is_8bit();
  const bool is_1bit = quantized_points.is_1bit();
  const size_t N = hn::Lanes(i8);
  const size_t kPoints = hn::Lanes(i32);
  const size_t num_points = indices.size();
  const size_t bytes_per_point = quantized_points.num_bytes_per_datapoint();
  const size_t dims = quantized_points.dimensionality();
  const size_t num_dim_chunks = hwy::DivCeil(bytes_per_point, N);
  const size_t num_hamming_tiles = hwy::DivCeil(bytes_per_point, 4);
  const size_t tiles_per_chunk = is_8bit ? N / 4 : 2 * N;
  const size_t total_tiles = is_1bit ? num_hamming_tiles
                                     : num_dim_chunks * tiles_per_chunk;
  const size_t panel_bytes = total_tiles * N;
  const size_t decoded_dim = is_8bit ? num_dim_chunks * N
                                     : num_dim_chunks * 8 * N;

  ProcessLeafInitImpl(quantized_points, indices, workspace);

  const size_t stride = quantized_points.stride();
  const uint8_t* raw = quantized_points.quantized_data();

  using Float = hn::ScalableTag<float>;
  const Float f32;
  const float inf = std::numeric_limits<float>::max();

  workspace->fused_topk_scores.assign(num_points * K, inf);
  workspace->fused_topk_indices.assign(num_points * K, -1);

  const size_t soa_size = hwy::RoundUpTo(num_points + kPoints, kPoints);
  workspace->fused_db_s.resize(K * soa_size);
  workspace->fused_db_idx_s.resize(K * soa_size);
  const auto inf_fill = hn::Set(f32, inf);
  const auto neg1_fill = hn::Set(i32, -1);
  for (size_t off = 0; off < K * soa_size; off += kPoints) {
    hn::Store(inf_fill, f32, workspace->fused_db_s.data() + off);
    hn::Store(neg1_fill, i32, workspace->fused_db_idx_s.data() + off);
  }

  if (!is_1bit) workspace->all_q_decoded.resize(kMqFull * decoded_dim);

  for (size_t q_start = 0; q_start < num_points; q_start += kMqFull) {
    const size_t actual_mq = std::min(kMqFull, num_points - q_start);

    if (!is_1bit) {
      if (is_8bit) {
        DecodeQueries8BitNoSums(u8, i8, raw, stride, actual_mq, bytes_per_point,
                                num_dim_chunks, N,
                                workspace->all_q_decoded.data(),
                                indices.data() + q_start);
      }
    }

    const int8_t* query_ptrs[kMqFull];
    const uint8_t* hamming_query_ptrs[kMqFull];
    float q_norms[kMqFull];
    size_t q_abs_indices[kMqFull];
    hn::Vec<Int32> neg_correction[kMqFull];
    for (size_t q = 0; q < actual_mq; ++q) {
      if (is_1bit) {
        hamming_query_ptrs[q] = raw + indices[q_start + q] * stride;
      } else {
        query_ptrs[q] = workspace->all_q_decoded.data() + q * decoded_dim;
      }
      q_norms[q] = -workspace->norms[q_start + q];
      neg_correction[q] =
          hn::Set(i32, -128 * workspace->all_q_byte_sums[q_start + q]);
      q_abs_indices[q] = q_start + q;
    }
    for (size_t q = actual_mq; q < kMqFull; ++q) {
      if (is_1bit) {
        hamming_query_ptrs[q] = hamming_query_ptrs[actual_mq - 1];
      } else {
        query_ptrs[q] = query_ptrs[actual_mq - 1];
      }
      q_norms[q] = 0.0f;
      neg_correction[q] = hn::Zero(i32);
      q_abs_indices[q] = q_abs_indices[actual_mq - 1];
    }

    const auto inf_v = hn::Set(f32, inf);
    const auto lane_idx = hn::Iota(i32, 0);
    hn::Vec<Float> minv[kMqFull * K];
    hn::Vec<Int32> minidx[kMqFull * K];
    for (size_t t = 0; t < kMqFull * K; ++t) {
      minv[t] = inf_v;
      minidx[t] = hn::Set(i32, -1);
    }

    const size_t qbuf_tile_stride = num_hamming_tiles * N;
    HWY_ALIGN uint8_t hamming_qbuf[kMqFull * 64 * 64];
    if (is_1bit) {
      PreBroadcastHammingQueries<kMqFull>(u8, i32, hamming_query_ptrs,
          num_hamming_tiles, bytes_per_point, N, hamming_qbuf);
    }

    const size_t max_db = std::min(q_start + actual_mq, num_points);
    const size_t panels_needed = hwy::DivCeil(max_db, kPoints);

    auto run_epilogue = [&](hn::Vec<Int32>* acc, size_t base, size_t kP) {
      const auto neg_db_norms_v =
          hn::LoadN(f32, workspace->norms.data() + base, kP);
      hn::Vec<Float> dm[K];
      hn::Vec<Int32> di[K];
      for (size_t k = 0; k < K; ++k) {
        dm[k] = hn::Load(f32, workspace->fused_db_s.data() + k * soa_size + base);
        di[k] = hn::Load(i32, workspace->fused_db_idx_s.data() + k * soa_size + base);
      }
      if (is_1bit) {
        ScoreAndScatterEpilogueK<K,true>(
            f32, i32, acc, actual_mq, neg_correction, neg_db_norms_v, q_norms,
            q_abs_indices, base, kP, kPoints, lane_idx, inf_v, minv, minidx, dm,
            di, dims);
      } else {
        ScoreAndScatterEpilogueK<K>(
            f32, i32, acc, actual_mq, neg_correction, neg_db_norms_v, q_norms,
            q_abs_indices, base, kP, kPoints, lane_idx, inf_v, minv, minidx, dm,
            di);
      }
      for (size_t k = 0; k < K; ++k) {
        hn::Store(dm[k], f32, workspace->fused_db_s.data() + k * soa_size + base);
        hn::Store(di[k], i32, workspace->fused_db_idx_s.data() + k * soa_size + base);
      }
    };

    size_t p = 0;
    for (; p + 4 <= panels_needed; p += 4) {
      hn::Vec<Int32> acc0[kMqFull], acc1[kMqFull], acc2[kMqFull], acc3[kMqFull];
      if (is_1bit) {
        HammingMicroKernelAccumulate4<kMqFull>(
            u8, i8, i32, hamming_qbuf, qbuf_tile_stride,
            workspace->panel_data.data() + p * panel_bytes,
            workspace->panel_data.data() + (p + 1) * panel_bytes,
            workspace->panel_data.data() + (p + 2) * panel_bytes,
            workspace->panel_data.data() + (p + 3) * panel_bytes,
            num_hamming_tiles, N, acc0, acc1, acc2, acc3);
      } else {
        MicroKernelAccumulate4<kMqFull>(
            u8, i8, i32, query_ptrs,
            workspace->panel_data.data() + p * panel_bytes,
            workspace->panel_data.data() + (p + 1) * panel_bytes,
            workspace->panel_data.data() + (p + 2) * panel_bytes,
            workspace->panel_data.data() + (p + 3) * panel_bytes,
            total_tiles, N, acc0, acc1, acc2, acc3);
      }
      for (int pi = 0; pi < 4; ++pi) {
        const size_t base = (p + pi) * kPoints;
        const size_t kP = std::min(kPoints, num_points - base);
        auto* acc = pi == 0 ? acc0 : pi == 1 ? acc1 : pi == 2 ? acc2 : acc3;
        run_epilogue(acc, base, kP);
      }
    }
    for (; p + 2 <= panels_needed; p += 2) {
      hn::Vec<Int32> acc0[kMqFull], acc1[kMqFull];
      if (is_1bit) {
        HammingMicroKernelAccumulate<kMqFull>(
            u8, i8, i32, hamming_qbuf, qbuf_tile_stride,
            workspace->panel_data.data() + p * panel_bytes,
            workspace->panel_data.data() + (p + 1) * panel_bytes,
            num_hamming_tiles, N, acc0, acc1);
      } else {
        MicroKernelAccumulate<kMqFull>(
            u8, i8, i32, query_ptrs,
            workspace->panel_data.data() + p * panel_bytes,
            workspace->panel_data.data() + (p + 1) * panel_bytes,
            total_tiles, N, acc0, acc1);
      }
      for (int pi = 0; pi < 2; ++pi) {
        const size_t base = (p + pi) * kPoints;
        const size_t kP = std::min(kPoints, num_points - base);
        auto* acc = pi == 0 ? acc0 : acc1;
        run_epilogue(acc, base, kP);
      }
    }
    for (; p < panels_needed; ++p) {
      const size_t base = p * kPoints;
      const size_t kP = std::min(kPoints, num_points - base);
      hn::Vec<Int32> acc[kMqFull];
      if (is_1bit) {
        HammingMicroKernelAccumulate1<kMqFull>(
            u8, i8, i32, hamming_qbuf, qbuf_tile_stride,
            workspace->panel_data.data() + p * panel_bytes,
            num_hamming_tiles, N, acc);
      } else {
        MicroKernelAccumulate1<kMqFull>(
            u8, i8, i32, query_ptrs,
            workspace->panel_data.data() + p * panel_bytes,
            total_tiles, N, acc);
      }
      run_epilogue(acc, base, kP);
    }

    HWY_ALIGN float cand_s[16 * 64];
    HWY_ALIGN int32_t cand_i[16 * 64];
    for (size_t q = 0; q < actual_mq; ++q) {
      const size_t qi = q_abs_indices[q];
      const float qn = q_norms[q];
      for (size_t k = 0; k < K; ++k) {
        hn::Store(minv[q * K + k], f32, cand_s + k * kPoints);
        hn::Store(minidx[q * K + k], i32, cand_i + k * kPoints);
      }
      float* outs = workspace->fused_topk_scores.data() + qi * K;
      int32_t* outi = workspace->fused_topk_indices.data() + qi * K;
      for (size_t c = 0; c < K * kPoints; ++c) {
        if (cand_i[c] < 0) continue;
        ScalarInsertTopK(outs, outi, K, cand_s[c] * qn, cand_i[c]);
      }
    }
  }

  for (size_t i = 0; i < num_points; ++i) {
    float* outs = workspace->fused_topk_scores.data() + i * K;
    int32_t* outi = workspace->fused_topk_indices.data() + i * K;
    for (size_t k = 0; k < K; ++k) {
      ScalarInsertTopK(outs, outi, K, workspace->fused_db_s[k * soa_size + i],
                       workspace->fused_db_idx_s[k * soa_size + i]);
    }
  }
}

}
}

}
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace pipnn {
void ProcessLeafTopKFused(const RsqBasePoints& quantized_points,
                          absl::Span<const int> indices, size_t k,
                          ProcessLeafWorkspace* workspace) {
#define PIPNN_FUSED_CASE(KK)                                            \
  case KK:                                                             \
    HWY_STATIC_DISPATCH(ProcessLeafTopKFusedImpl<KK>)                  \
    (quantized_points, indices, workspace);                           \
    break;
  switch (k) {
    PIPNN_FUSED_CASE(3)
    PIPNN_FUSED_CASE(4)
    PIPNN_FUSED_CASE(5)
    PIPNN_FUSED_CASE(6)
    PIPNN_FUSED_CASE(7)
    PIPNN_FUSED_CASE(8)
    PIPNN_FUSED_CASE(9)
    PIPNN_FUSED_CASE(10)
    PIPNN_FUSED_CASE(11)
    PIPNN_FUSED_CASE(12)
    PIPNN_FUSED_CASE(13)
    PIPNN_FUSED_CASE(14)
    PIPNN_FUSED_CASE(15)
    default:

      std::abort();
  }
#undef PIPNN_FUSED_CASE
}

}

#endif

