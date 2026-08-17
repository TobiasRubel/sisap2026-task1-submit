#ifndef PIPNN_QUANTIZATION_ROTATION_H_
#define PIPNN_QUANTIZATION_ROTATION_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>

namespace pipnn {
namespace internal {
void fwht_inplace(float* data, size_t n);

}

class HadamardAndMatrixRotation {
 public:
  HadamardAndMatrixRotation(size_t dimensionality, size_t seed)
      : dim_(dimensionality) {
    QCHECK_GT(dim_, size_t{0});

    num_subspaces_ = dim_;
    while (num_subspaces_ > 1 && num_subspaces_ % 2 == 0) {
      num_subspaces_ /= 2;
    }
    subspace_dim_ = dim_ / num_subspaces_;

    const int rs = static_cast<int>(num_subspaces_);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Eigen::MatrixXf rand_mat(rs, rs);
    for (int i = 0; i < rs; ++i)
      for (int j = 0; j < rs; ++j)
        rand_mat(i, j) = dist(gen);
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(rand_mat);
    rotation_matrix_ =
        std::sqrt(static_cast<float>(rs)) *
        (qr.householderQ() * Eigen::MatrixXf::Identity(rs, rs));
  }

  absl::Status Rotate(absl::Span<float> input_coordinates,
                      absl::Span<float> output_coordinates) const {
    if (input_coordinates.size() != dim_ ||
        output_coordinates.size() != dim_) {
      return absl::InvalidArgumentError(
          "'input_coordinates' and 'output_coordinates' must have the same "
          "dimensionality as specified in the constructor.");
    }

    if (subspace_dim_ > 1) {
      for (size_t i = 0; i < dim_; i += subspace_dim_) {
        internal::fwht_inplace(input_coordinates.data() + i, subspace_dim_);
      }
    }

    const Eigen::Map<const Eigen::MatrixXf, Eigen::Unaligned> in_map(
        input_coordinates.data(), subspace_dim_, num_subspaces_);
    Eigen::Map<Eigen::MatrixXf, Eigen::Unaligned> out_map(
        output_coordinates.data(), subspace_dim_, num_subspaces_);
    out_map.noalias() = in_map * rotation_matrix_;

    return absl::OkStatus();
  }

 private:
  size_t dim_;
  size_t num_subspaces_;
  size_t subspace_dim_;
  Eigen::MatrixXf rotation_matrix_;
};

}

#endif
