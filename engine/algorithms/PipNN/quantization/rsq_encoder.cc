#include "rsq_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "parlay/parallel.h"

#define RETURN_IF_ERROR(expr)                        \
  do {                                               \
    auto _status = (expr);                           \
    if (!_status.ok()) return _status;               \
  } while (0)

#define ASSIGN_OR_RETURN_IMPL_(statusor, lhs, rhs)   \
  auto statusor = (rhs);                             \
  if (!statusor.ok()) return statusor.status();      \
  lhs = std::move(statusor).value()

#define ASSIGN_OR_RETURN(lhs, rhs)                   \
  ASSIGN_OR_RETURN_IMPL_(                            \
      HWY_CONCAT(status_or_tmp_, __COUNTER__), lhs, rhs)

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rsq_encoder.cc"

#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace pipnn {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void FlipSignsAndScaleImpl(const float* HWY_RESTRICT coordinates,
                           const float* HWY_RESTRICT signs, float inv_norm,
                           size_t dimensionality, float* HWY_RESTRICT output) {
  const hn::ScalableTag<float> f32;
  const size_t N = hn::Lanes(f32);
  const auto inv_norm_v = hn::Set(f32, inv_norm);
  size_t i = 0;
  for (; i + N <= dimensionality; i += N) {
    const auto c = hn::LoadU(f32, coordinates + i);
    const auto s = hn::LoadU(f32, signs + i);
    hn::StoreU(hn::Mul(hn::Mul(c, s), inv_norm_v), f32, output + i);
  }
  for (; i < dimensionality; ++i) {
    output[i] = coordinates[i] * signs[i] * inv_norm;
  }
}

template <class DBlock>
HWY_INLINE void FwhtInVectorStage(DBlock,
                                   float* HWY_RESTRICT data, size_t n) {
  const hn::Half<DBlock> dh;
  const size_t Nh = hn::Lanes(dh);
  for (size_t blk = 0; blk + 2 * Nh <= n; blk += 2 * Nh) {
    const auto lo = hn::LoadU(dh, data + blk);
    const auto hi = hn::LoadU(dh, data + blk + Nh);
    hn::StoreU(hn::Add(lo, hi), dh, data + blk);
    hn::StoreU(hn::Sub(lo, hi), dh, data + blk + Nh);
  }
}

void FwhtInPlaceImpl(float* HWY_RESTRICT data, size_t n) {
  if (n < 2) return;
  const hn::ScalableTag<float> d;
  const size_t N = hn::Lanes(d);

  size_t hbs = n >> 1;
  for (; hbs >= N; hbs >>= 1) {
    for (size_t blk = 0; blk < n; blk += 2 * hbs) {
      for (size_t j = 0; j < hbs; j += N) {
        const auto a = hn::LoadU(d, data + blk + j);
        const auto b = hn::LoadU(d, data + blk + hbs + j);
        hn::StoreU(hn::Add(a, b), d, data + blk + j);
        hn::StoreU(hn::Sub(a, b), d, data + blk + hbs + j);
      }
    }
  }

  if (N >= 16 && hbs == 8) {
    FwhtInVectorStage(hn::CappedTag<float, 16>(), data, n);
    hbs >>= 1;
  }
  if (N >= 8 && hbs == 4) {
    FwhtInVectorStage(hn::CappedTag<float, 8>(), data, n);
    hbs >>= 1;
  }
  if (N >= 4 && hbs == 2) {
    FwhtInVectorStage(hn::CappedTag<float, 4>(), data, n);
    hbs >>= 1;
  }
  if (N >= 2 && hbs == 1) {
    FwhtInVectorStage(hn::CappedTag<float, 2>(), data, n);
    hbs >>= 1;
  }
}

float EightBitEncodeImpl(const float* HWY_RESTRICT coordinates,
                         size_t dimensionality,
                         uint8_t* HWY_RESTRICT output) {
  const hn::ScalableTag<float> f32;
  const hn::ScalableTag<int32_t> i32;
  const size_t NF = hn::Lanes(f32);

  auto max_abs_v = hn::Zero(f32);
  size_t d = 0;
  for (; d + NF <= dimensionality; d += NF) {
    max_abs_v = hn::Max(max_abs_v, hn::Abs(hn::LoadU(f32, coordinates + d)));
  }
  float max_abs = hn::ReduceMax(f32, max_abs_v);
  for (; d < dimensionality; ++d) {
    max_abs = std::max(max_abs, std::abs(coordinates[d]));
  }
  if (max_abs == 0.0f) {
    std::memset(output, 0, dimensionality);
    return 0.0f;
  }

  const auto scale = hn::Set(f32, 127.0f / max_abs);
  const auto neg_cap = hn::Set(f32, -127.0f);
  const auto pos_cap = hn::Set(f32, 127.0f);

  auto sqr_norm_acc = hn::Zero(f32);
  d = 0;
  for (; d + NF <= dimensionality; d += NF) {
    auto v = hn::LoadU(f32, coordinates + d);
    v = hn::Mul(v, scale);
    v = hn::Clamp(v, neg_cap, pos_cap);
    const auto iv = hn::NearestInt(v);
    sqr_norm_acc = hn::MulAdd(hn::ConvertTo(f32, iv), hn::ConvertTo(f32, iv),
                              sqr_norm_acc);
    HWY_ALIGN int32_t tmp[HWY_MAX_LANES_D(hn::ScalableTag<int32_t>)];
    hn::Store(iv, i32, tmp);
    for (size_t k = 0; k < NF; ++k) {
      output[d + k] = static_cast<uint8_t>(static_cast<int8_t>(tmp[k]));
    }
  }

  float sqr_norm = hn::ReduceSum(f32, sqr_norm_acc);
  const float scalar_scale = 127.0f / max_abs;
  for (; d < dimensionality; ++d) {
    int8_t iv = static_cast<int8_t>(
        std::clamp(std::round(coordinates[d] * scalar_scale), -127.0f, 127.0f));
    output[d] = static_cast<uint8_t>(iv);
    sqr_norm += static_cast<float>(iv) * static_cast<float>(iv);
  }
  return sqr_norm;
}

int32_t ComputeByteSum8BitImpl(const uint8_t* HWY_RESTRICT data,
                               size_t num_bytes) {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int8_t> i8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);
  const auto ones = hn::Set(u8, uint8_t{1});
  auto acc = hn::Zero(i32);
  size_t j = 0;
  for (; j + N <= num_bytes; j += N) {
    const auto v = hn::Load(i8, reinterpret_cast<const int8_t*>(data + j));
    acc = hn::SumOfMulQuadAccumulate(i32, ones, v, acc);
  }
  int32_t result = hn::ReduceSum(i32, acc);
  for (; j < num_bytes; ++j) {
    result += static_cast<int8_t>(data[j]);
  }
  return result;
}

float OneBitEncodeImpl(const float* HWY_RESTRICT coordinates,
                       size_t dimensionality,
                       uint8_t* HWY_RESTRICT output) {
  const size_t num_bytes = (dimensionality + 7) / 8;
  std::memset(output, 0, num_bytes);

  size_t d = 0;
  for (; d + 8 <= dimensionality; d += 8) {
    uint8_t byte = 0;
    for (int b = 0; b < 8; ++b) {
      if (coordinates[d + b] < 0.0f) byte |= (1u << b);
    }
    output[d / 8] = byte;
  }

  if (d < dimensionality) {
    uint8_t byte = 0;
    for (size_t b = 0; d + b < dimensionality; ++b) {
      if (coordinates[d + b] < 0.0f) byte |= (1u << b);
    }
    output[d / 8] = byte;
  }

  return static_cast<float>(dimensionality) * 16129.0f;
}

int32_t ComputeByteSum1BitImpl(const uint8_t* HWY_RESTRICT data,
                               size_t num_bytes, size_t dimensionality) {
  const hn::ScalableTag<uint8_t> u8;
  const hn::ScalableTag<int32_t> i32;
  const size_t N = hn::Lanes(u8);

  const auto ones_i8 = hn::BitCast(hn::ScalableTag<int8_t>{},
                                    hn::Set(u8, uint8_t{1}));
  auto acc = hn::Zero(i32);

  size_t j = 0;
  for (; j + N <= num_bytes; j += N) {
    const auto bytes = hn::LoadU(u8, data + j);
    const auto popcnt = hn::PopulationCount(bytes);
    acc = hn::SumOfMulQuadAccumulate(i32, popcnt, ones_i8, acc);
  }

  int32_t total_popcount = hn::ReduceSum(i32, acc);
  for (; j < num_bytes; ++j) {
    total_popcount += __builtin_popcount(data[j]);
  }

  return 127 * (static_cast<int32_t>(dimensionality) - 2 * total_popcount);
}

}
}
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace pipnn {
namespace internal {
void fwht_inplace(float* data, size_t n) {
  HWY_STATIC_DISPATCH(FwhtInPlaceImpl)(data, n);
}
}

static size_t ComputeBytesPerDatapoint(size_t dims, QuantizationScheme scheme) {
  if (scheme == QuantizationScheme::kEightBit) return dims;
  return (dims + 7) / 8;
}

RsqEncoder::RsqEncoder(size_t dimensionality, size_t seed,
                           QuantizationScheme scheme)
    : dimensionality_(dimensionality),
      num_bytes_per_datapoint_(ComputeBytesPerDatapoint(dimensionality_, scheme)),
      signs_(dimensionality),
      rotation_(dimensionality, seed),
      scheme_(scheme) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> dist(0, 1);
  for (int i = 0; i < dimensionality; ++i) {
    int f = dist(gen);
    signs_[i] = (2.0f * f - 1);
  }
}

RsqBasePoints::RsqBasePoints(
    size_t dimensionality, size_t num_datapoints,
    bool pad_to_cache_line_size, QuantizationScheme scheme)
    : dimensionality_(dimensionality),
      num_datapoints_(num_datapoints),
      num_bytes_per_datapoint_(ComputeBytesPerDatapoint(dimensionality_, scheme)),
      stride_(pad_to_cache_line_size ? hwy::RoundUpTo(num_bytes_per_datapoint_,
                                                      ABSL_CACHELINE_SIZE)
                                     : num_bytes_per_datapoint_),
      quantized_data_(num_datapoints == 0
                          ? 0
                          : (num_datapoints - 1) * stride_ +
                                hwy::RoundUpTo(num_bytes_per_datapoint_,
                                               ABSL_CACHELINE_SIZE)),
      norm_data_(num_datapoints),
      scheme_(scheme) {}

absl::Status RsqEncoder::FlipRandomSignsNormalizeAndRotate(
    absl::Span<const float> coordinates, float norm,
    absl::Span<float> rotated_coordinates) const {
  if (norm == 0.0f) {
    return absl::InvalidArgumentError("Vector has a norm of zero.");
  }
  const float inv_norm = 1.0f / norm;

  static thread_local std::vector<float> random_sign_coordinates;
  random_sign_coordinates.resize(dimensionality_);

  HWY_STATIC_DISPATCH(FlipSignsAndScaleImpl)(coordinates.data(), signs_.data(),
                                              inv_norm, dimensionality_,
                                              random_sign_coordinates.data());
  return rotation_.Rotate(absl::MakeSpan(random_sign_coordinates),
                          rotated_coordinates);
}

absl::StatusOr<std::pair<float, float>>
RsqEncoder::EncodeDatapoint(
    absl::Span<const float> coordinates,
    absl::Span<uint8_t> output_coordinates) const {
  if (coordinates.size() != dimensionality_) {
    return absl::InvalidArgumentError(
        "Datapoint dimensionality does not match dimensionality supplied "
        "to RsqEncoder.");
  }
  if (output_coordinates.size() != num_bytes_per_datapoint_) {
    return absl::InvalidArgumentError(
        "Output coordinates size does not match number of bytes per "
        "datapoint.");
  }

  const float sqr_norm =
      Eigen::Map<Eigen::VectorXf>(const_cast<float*>(coordinates.data()),
                                  coordinates.size())
          .squaredNorm();

  if (!std::isfinite(sqr_norm)) {
    return absl::InvalidArgumentError("Infinite norm");
  }
  if (sqr_norm == 0.0f) {
    std::fill(output_coordinates.begin(), output_coordinates.end(), 0);
    return std::make_pair(0.0f, 0.0f);
  }
  const float norm = std::sqrt(sqr_norm);

  static thread_local std::vector<float> rotated_coordinates;
  rotated_coordinates.resize(dimensionality_);
  RETURN_IF_ERROR(FlipRandomSignsNormalizeAndRotate(
      coordinates, norm, absl::MakeSpan(rotated_coordinates)));

  if (scheme_ == QuantizationScheme::kEightBit) {
    const float quantized_sqr_norm = HWY_STATIC_DISPATCH(EightBitEncodeImpl)(
        rotated_coordinates.data(), dimensionality_, output_coordinates.data());
    const float scale = norm / std::sqrt(quantized_sqr_norm);
    return std::make_pair(sqr_norm, scale);
  }

  const float quantized_sqr_norm = HWY_STATIC_DISPATCH(OneBitEncodeImpl)(
      rotated_coordinates.data(), dimensionality_, output_coordinates.data());
  if (quantized_sqr_norm == 0.0f) {
    return std::make_pair(sqr_norm, 0.0f);
  }
  const float scale = norm / std::sqrt(quantized_sqr_norm);
  return std::make_pair(sqr_norm, scale);
}

absl::StatusOr<RsqBasePoints>
RsqBasePoints::Create(
    const RsqEncoder& rsq,
    absl::Span<const absl::Span<const float>> dense_datapoints,
    bool pad_to_cache_line_size) {
  if (dense_datapoints.empty()) {
    return absl::InvalidArgumentError("No datapoints provided.");
  }
  ASSIGN_OR_RETURN(auto base_points,
                   RsqBasePoints::Create(
                       rsq, dense_datapoints.size(), pad_to_cache_line_size));
  parlay::parallel_for(0, dense_datapoints.size(), [&](size_t i) {
    auto status = base_points.UpdatePoint(rsq, dense_datapoints[i], i);
    (void)status;
  }, 1000);
  return base_points;
}

absl::StatusOr<RsqBasePoints>
RsqBasePoints::Create(const RsqEncoder& rsq,
                      size_t num_datapoints,
                      bool pad_to_cache_line_size) {
  return RsqBasePoints(rsq.dimensionality(), num_datapoints,
                       pad_to_cache_line_size, rsq.scheme());
}

absl::Status RsqBasePoints::UpdatePoint(
    const RsqEncoder& rsq, absl::Span<const float> coordinates,
    size_t index) {
  if (index >= num_datapoints_) {
    return absl::InvalidArgumentError(
        "Index is larger than the number of datapoints.");
  }

  auto encode_status = rsq.EncodeDatapoint(
      coordinates,
      absl::MakeSpan(quantized_data_).subspan(index * stride_, num_bytes_per_datapoint_));
  if (!encode_status.ok()) return encode_status.status();
  auto [sqr_norm, scale] = std::move(encode_status).value();
  (void)sqr_norm;
  norm_data_.Set(index, scale);

  const uint8_t* qd = quantized_data_.data() + index * stride_;
  if (scheme_ == QuantizationScheme::kEightBit) {
    norm_data_.SetByteSum(index, HWY_STATIC_DISPATCH(ComputeByteSum8BitImpl)(
                                     qd, num_bytes_per_datapoint_));
  } else {
    norm_data_.SetByteSum(index, HWY_STATIC_DISPATCH(ComputeByteSum1BitImpl)(
                                     qd, num_bytes_per_datapoint_,
                                     dimensionality_));
  }
  return absl::OkStatus();
}

absl::Status RsqBasePoints::UpdatePoint(
    const RsqBasePointView& datapoint, size_t index) {
  if (index >= num_datapoints_) {
    return absl::InvalidArgumentError(
        "Index is larger than the number of datapoints.");
  }
  if (datapoint.quantized_data.size() != num_bytes_per_datapoint_) {
    return absl::InvalidArgumentError(
        "Quantized datapoint size does not match number of bytes per "
        "datapoint.");
  }
  norm_data_.Set(index, datapoint.norm_scaling_factor);
  std::memcpy(quantized_data_.data() + index * stride_,
              datapoint.quantized_data.data(), num_bytes_per_datapoint_);

  const uint8_t* qd = quantized_data_.data() + index * stride_;
  if (scheme_ == QuantizationScheme::kEightBit) {
    norm_data_.SetByteSum(index, HWY_STATIC_DISPATCH(ComputeByteSum8BitImpl)(
                                     qd, num_bytes_per_datapoint_));
  } else {
    norm_data_.SetByteSum(index, HWY_STATIC_DISPATCH(ComputeByteSum1BitImpl)(
                                     qd, num_bytes_per_datapoint_,
                                     dimensionality_));
  }
  return absl::OkStatus();
}

}

#endif
