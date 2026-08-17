#ifndef PIPNN_QUANTIZATION_PIPNN_RSQ_H_
#define PIPNN_QUANTIZATION_PIPNN_RSQ_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "rsq_encoder.h"
#include "absl/types/span.h"
#include "hwy/aligned_allocator.h"

namespace pipnn {
class PreTransposedRsqDataset {
 public:
  static PreTransposedRsqDataset Build(
      const RsqBasePoints& base_points,
      absl::Span<const int> indices);

  size_t num_points() const { return num_points_; }
  size_t num_panels() const { return num_panels_; }
  size_t panel_bytes() const { return panel_bytes_; }
  const uint8_t* panel_data() const { return panel_data_.data(); }
  const float* norms() const { return norms_.data(); }

  size_t num_points_ = 0;
  size_t num_panels_ = 0;
  size_t panel_bytes_ = 0;
  hwy::AlignedVector<uint8_t> panel_data_;
  hwy::AlignedVector<float> norms_;
};

struct ProcessLeafWorkspace {
  hwy::AlignedVector<int8_t> all_q_decoded;
  std::vector<int32_t> all_q_byte_sums;
  hwy::AlignedVector<uint8_t> panel_data;
  hwy::AlignedVector<float> norms;

  hwy::AlignedVector<float> fused_topk_scores;
  hwy::AlignedVector<int32_t> fused_topk_indices;
  hwy::AlignedVector<float> fused_db_s;
  hwy::AlignedVector<int32_t> fused_db_idx_s;
};

void ProcessLeafTopKFused(const RsqBasePoints& quantized_points,
                          absl::Span<const int> indices, size_t k,
                          ProcessLeafWorkspace* workspace);

template <size_t K>
struct TopKResult {
  float scores[K];
  int32_t indices[K];

  TopKResult() {
    for (size_t i = 0; i < K; ++i) {
      scores[i] = std::numeric_limits<float>::max();
      indices[i] = -1;
    }
  }
};

struct AssignToLeadersWorkspace {
  hwy::AlignedVector<int8_t> all_q_decoded;
  std::vector<int32_t> all_q_byte_sums;
  std::vector<float> q_norms;

  hwy::AlignedVector<float> min_scores;
  hwy::AlignedVector<int32_t> min_indices;
};

void AssignToLeadersTop1(const RsqBasePoints& quantized_points,
                         absl::Span<const int> query_indices,
                         const PreTransposedRsqDataset& leaders,
                         std::vector<TopKResult<1>>& top1_results,
                         AssignToLeadersWorkspace* workspace = nullptr);

template <size_t K>
void AssignToLeadersTopK(const RsqBasePoints& quantized_points,
                         absl::Span<const int> query_indices,
                         const PreTransposedRsqDataset& leaders,
                         std::vector<TopKResult<K>>& topk_results,
                         AssignToLeadersWorkspace* workspace = nullptr);

template <size_t K>
void AssignToLeadersConsecutiveTopK(
    const RsqBasePoints& quantized_points, size_t query_begin,
    size_t query_end, const PreTransposedRsqDataset& leaders,
    std::vector<TopKResult<K>>& topk_results,
    AssignToLeadersWorkspace* workspace = nullptr);

}

#endif
