#ifndef PIPNN_QUANTIZATION_RSQ_ENCODER_H_
#define PIPNN_QUANTIZATION_RSQ_ENCODER_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "rotation.h"
#include "hwy/aligned_allocator.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pipnn {
template <typename T>
struct NoInitAllocator : hwy::AlignedAllocator<T> {
  using hwy::AlignedAllocator<T>::AlignedAllocator;
  void construct(T*) noexcept {}
  template <typename Arg, typename... Args>
  void construct(T* p, Arg&& arg, Args&&... args) {
    new (p) T(std::forward<Arg>(arg), std::forward<Args>(args)...);
  }
};

enum class QuantizationScheme { kEightBit, kOneBit };

class RsqEncoder {
 public:
  RsqEncoder(size_t dimensionality, size_t seed,
               QuantizationScheme scheme = QuantizationScheme::kEightBit);

  absl::StatusOr<std::pair<float, float>> EncodeDatapoint(
      absl::Span<const float> coordinates,
      absl::Span<uint8_t> output_coordinates) const;

  size_t dimensionality() const { return dimensionality_; }
  size_t num_bytes_per_datapoint() const { return num_bytes_per_datapoint_; }
  QuantizationScheme scheme() const { return scheme_; }
  bool is_8bit() const { return scheme_ == QuantizationScheme::kEightBit; }
  bool is_1bit() const { return scheme_ == QuantizationScheme::kOneBit; }

 protected:
  absl::Status FlipRandomSignsNormalizeAndRotate(
      absl::Span<const float> coordinates, float norm,
      absl::Span<float> rotated_coordinates) const;

  size_t dimensionality_ = 0;
  size_t num_bytes_per_datapoint_ = 0;
  std::vector<float> signs_;
  HadamardAndMatrixRotation rotation_;
  QuantizationScheme scheme_ = QuantizationScheme::kEightBit;
};

struct RsqBasePointView {
  absl::Span<const uint8_t> quantized_data;
  const float norm_scaling_factor = 0.0f;
};

namespace internal {
class NormVector {
 public:
  explicit NormVector(size_t num_datapoints) {
    data_.resize(num_datapoints * 2);
  }

  void Set(size_t index, float norm_scaling_factor) {
    data_[index * 2] = norm_scaling_factor;
  }

  float GetNormScalingFactor(size_t index) const { return data_[index * 2]; }

  void SetByteSum(size_t index, int32_t byte_sum) {
    *reinterpret_cast<int32_t*>(&data_[index * 2 + 1]) = byte_sum;
  }

  int32_t GetByteSum(size_t index) const {
    return *reinterpret_cast<const int32_t*>(&data_[index * 2 + 1]);
  }

 private:
  std::vector<float> data_;
};

}

class RsqBasePoints {
 public:
  static absl::StatusOr<RsqBasePoints> Create(
      const RsqEncoder& rsq,
      absl::Span<const absl::Span<const float>> dense_datapoints,
      bool pad_to_cache_line_size = false);

  static absl::StatusOr<RsqBasePoints> Create(
      const RsqEncoder& rsq, size_t num_datapoints,
      bool pad_to_cache_line_size = false);

  absl::Status UpdatePoint(const RsqEncoder& rsq,
                           absl::Span<const float> coordinates, size_t index);

  absl::Status UpdatePoint(const RsqBasePointView& datapoint, size_t index);

  size_t num_datapoints() const { return num_datapoints_; }
  size_t num_bytes_per_datapoint() const { return num_bytes_per_datapoint_; }
  size_t stride() const { return stride_; }
  size_t dimensionality() const { return dimensionality_; }
  const uint8_t* quantized_data() const { return quantized_data_.data(); }
  size_t quantized_data_size() const { return quantized_data_.size(); }
  QuantizationScheme scheme() const { return scheme_; }
  bool is_8bit() const { return scheme_ == QuantizationScheme::kEightBit; }
  bool is_1bit() const { return scheme_ == QuantizationScheme::kOneBit; }

  float norm_scaling_factor(size_t index) const {
    return norm_data_.GetNormScalingFactor(index);
  }

  int32_t byte_sum(size_t index) const { return norm_data_.GetByteSum(index); }

  const uint8_t* GetQuantizedDataPtr(size_t index) const {
    return quantized_data_.data() + index * stride_;
  }

 private:
  size_t dimensionality_ = 0;
  size_t num_datapoints_ = 0;
  size_t num_bytes_per_datapoint_ = 0;
  size_t stride_ = 0;
  std::vector<uint8_t, NoInitAllocator<uint8_t>> quantized_data_;
  internal::NormVector norm_data_;
  QuantizationScheme scheme_ = QuantizationScheme::kEightBit;

  RsqBasePoints(size_t dimensionality, size_t num_datapoints,
                bool pad_to_cache_line_size, QuantizationScheme scheme);
};

}

#endif
