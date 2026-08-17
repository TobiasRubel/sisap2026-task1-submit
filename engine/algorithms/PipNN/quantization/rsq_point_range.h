#pragma once

#include "rsq_encoder.h"

namespace pipnn {
struct RsqPointRange {
  RsqBasePoints base_points;

  explicit RsqPointRange(RsqBasePoints bp) : base_points(std::move(bp)) {}
};

}
