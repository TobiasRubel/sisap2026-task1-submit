#include "pipnn_rsq.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "rsq_encoder.h"
#include "absl/base/prefetch.h"
#include "absl/types/span.h"
#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "pipnn_rsq.cc"

#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace pipnn {
namespace HWY_NAMESPACE {
namespace {
namespace hn = hwy::HWY_NAMESPACE;

#include "pipnn_rsq-inl.h"

PreTransposedRsqDataset BuildPreTransposed(
    const RsqBasePoints& base_points,
    absl::Span<const int> indices) {
  using Uint8 = hn::ScalableTag<uint8_t>;
  const Uint8 u8;
  const size_t N = hn::Lanes(u8);
  const size_t kPoints = N / 4;

  const bool is_8bit = base_points.is_8bit();
  const bool is_1bit = base_points.is_1bit();
  const size_t num_points = indices.size();
  const size_t bytes_per_point = base_points.num_bytes_per_datapoint();
  const size_t stride = base_points.stride();
  const size_t num_dim_chunks = hwy::DivCeil(bytes_per_point, N);

  const bool use_hamming = is_1bit;
  const size_t num_hamming_tiles = hwy::DivCeil(bytes_per_point, 4);
  const size_t tiles_per_chunk = is_1bit ? 2 * N : N / 4;
  const size_t total_tiles = use_hamming ? num_hamming_tiles
                                         : num_dim_chunks * tiles_per_chunk;
  const size_t panel_bytes = total_tiles * N;
  const size_t num_panels = hwy::DivCeil(num_points, kPoints);

  std::vector<uint8_t> gathered_data(num_points * stride + N, 0);
  const uint8_t* raw = base_points.quantized_data();
  for (size_t i = 0; i < num_points; ++i) {
    std::memcpy(gathered_data.data() + i * stride, raw + indices[i] * stride,
                stride);
  }

  hwy::AlignedVector<uint8_t> panel_data(num_panels * panel_bytes);
  if (is_8bit) {
    for (size_t p = 0; p < num_panels; ++p) {
      const size_t pi_start = p * kPoints;
      const size_t kP = std::min(kPoints, num_points - pi_start);
      DecodeDbPanel8Bit(u8, gathered_data.data(), stride, 0,
                        pi_start, kP, kPoints, bytes_per_point, num_dim_chunks,
                        tiles_per_chunk, N,
                        panel_data.data() + p * panel_bytes);
    }
  } else if (is_1bit) {
    for (size_t p = 0; p < num_panels; ++p) {
      const size_t pi_start = p * kPoints;
      const size_t kP = std::min(kPoints, num_points - pi_start);
      BuildHammingPanel1Bit(u8, gathered_data.data(), stride, 0, pi_start, kP,
                            kPoints, bytes_per_point, num_hamming_tiles, N,
                            panel_data.data() + p * panel_bytes);
    }
  }

  hwy::AlignedVector<float> norms(num_panels * kPoints);
  std::fill(norms.begin(), norms.end(),
            std::numeric_limits<float>::quiet_NaN());
  for (size_t i = 0; i < num_points; ++i) {
    norms[i] = base_points.norm_scaling_factor(indices[i]);
  }

  PreTransposedRsqDataset result;
  result.num_points_ = num_points;
  result.num_panels_ = num_panels;
  result.panel_bytes_ = panel_bytes;
  result.panel_data_ = std::move(panel_data);
  result.norms_ = std::move(norms);
  return result;
}

}
}

}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace pipnn {
HWY_EXPORT(BuildPreTransposed);

PreTransposedRsqDataset PreTransposedRsqDataset::Build(
    const RsqBasePoints& base_points,
    absl::Span<const int> indices) {
  return HWY_DYNAMIC_DISPATCH(BuildPreTransposed)(base_points, indices);
}

}

#endif
