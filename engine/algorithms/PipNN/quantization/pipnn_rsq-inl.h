#include <cstddef>
#include <cstdint>
#include <cstring>

#include "rsq_encoder.h"
#include "absl/base/prefetch.h"
#include "hwy/highway.h"

namespace hn = hwy::HWY_NAMESPACE;

template <size_t Mq>
HWY_INLINE void MicroKernelAccumulate(
    const hn::ScalableTag<uint8_t> u8, const hn::ScalableTag<int8_t> i8,
    const hn::ScalableTag<int32_t> i32,
    const int8_t* HWY_RESTRICT const* query_ptrs,
    const uint8_t* HWY_RESTRICT block0, const uint8_t* HWY_RESTRICT block1,
    size_t total_tiles, size_t N,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc0,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc1) {
  for (size_t q = 0; q < Mq; ++q) {
    acc0[q] = hn::Zero(i32);
    acc1[q] = hn::Zero(i32);
  }

  for (size_t t = 0; t < total_tiles; ++t) {
    const auto b0 = hn::Load(u8, block0 + t * N);
    const auto b1 = hn::Load(u8, block1 + t * N);
    for (size_t q = 0; q < Mq; ++q) {
      const auto qv = hn::BitCast(
          i8, hn::Set(i32, reinterpret_cast<const int32_t*>(query_ptrs[q])[t]));
      acc0[q] = hn::SumOfMulQuadAccumulate(i32, b0, qv, acc0[q]);
      acc1[q] = hn::SumOfMulQuadAccumulate(i32, b1, qv, acc1[q]);
    }
  }
}

template <size_t Mq>
HWY_INLINE void MicroKernelAccumulate1(
    const hn::ScalableTag<uint8_t> u8, const hn::ScalableTag<int8_t> i8,
    const hn::ScalableTag<int32_t> i32,
    const int8_t* HWY_RESTRICT const* query_ptrs,
    const uint8_t* HWY_RESTRICT block, size_t total_tiles, size_t N,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc) {
  for (size_t q = 0; q < Mq; ++q) acc[q] = hn::Zero(i32);

  for (size_t t = 0; t < total_tiles; ++t) {
    const auto b = hn::Load(u8, block + t * N);
    for (size_t q = 0; q < Mq; ++q) {
      const auto qv = hn::BitCast(
          i8, hn::Set(i32, reinterpret_cast<const int32_t*>(query_ptrs[q])[t]));
      acc[q] = hn::SumOfMulQuadAccumulate(i32, b, qv, acc[q]);
    }
  }
}

template <size_t Mq>
HWY_INLINE void MicroKernelAccumulate4(
    const hn::ScalableTag<uint8_t> u8, const hn::ScalableTag<int8_t> i8,
    const hn::ScalableTag<int32_t> i32,
    const int8_t* HWY_RESTRICT const* query_ptrs,
    const uint8_t* HWY_RESTRICT block0, const uint8_t* HWY_RESTRICT block1,
    const uint8_t* HWY_RESTRICT block2, const uint8_t* HWY_RESTRICT block3,
    size_t total_tiles, size_t N,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc0,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc1,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc2,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc3) {
  for (size_t q = 0; q < Mq; ++q) {
    acc0[q] = hn::Zero(i32);
    acc1[q] = hn::Zero(i32);
    acc2[q] = hn::Zero(i32);
    acc3[q] = hn::Zero(i32);
  }
  for (size_t t = 0; t < total_tiles; ++t) {
    const auto b0 = hn::Load(u8, block0 + t * N);
    const auto b1 = hn::Load(u8, block1 + t * N);
    const auto b2 = hn::Load(u8, block2 + t * N);
    const auto b3 = hn::Load(u8, block3 + t * N);
    for (size_t q = 0; q < Mq; ++q) {
      const auto qv = hn::BitCast(
          i8, hn::Set(i32, reinterpret_cast<const int32_t*>(query_ptrs[q])[t]));
      acc0[q] = hn::SumOfMulQuadAccumulate(i32, b0, qv, acc0[q]);
      acc1[q] = hn::SumOfMulQuadAccumulate(i32, b1, qv, acc1[q]);
      acc2[q] = hn::SumOfMulQuadAccumulate(i32, b2, qv, acc2[q]);
      acc3[q] = hn::SumOfMulQuadAccumulate(i32, b3, qv, acc3[q]);
    }
  }
}

template <typename Int8, typename Uint8>
HWY_INLINE void DecodeQueries8BitNoSums(
    const Uint8, const Int8 i8,
    const uint8_t* HWY_RESTRICT q_data,
    size_t q_stride, size_t num_queries, size_t bytes_per_point,
    size_t num_dim_chunks, size_t N, int8_t* HWY_RESTRICT all_q_decoded,
    const int* indices = nullptr) {
  const size_t decoded_dim = num_dim_chunks * N;

  for (size_t qi = 0; qi < num_queries; ++qi) {
    const size_t src = indices ? indices[qi] : qi;
    const uint8_t* q_base = q_data + src * q_stride;
    int8_t* q_out = all_q_decoded + qi * decoded_dim;
    size_t d = 0;
    for (; d + N <= bytes_per_point; d += N) {
      const auto v = hn::LoadU(i8, reinterpret_cast<const int8_t*>(q_base + d));
      hn::StoreU(v, i8, q_out + d);
    }

    for (size_t k = d; k < decoded_dim; ++k) {
      q_out[k] = (k < bytes_per_point)
                     ? static_cast<int8_t>(q_base[k])
                     : 0;
    }
  }
}

template <typename Int8, typename Uint8, typename Int32>
HWY_INLINE void DecodeQueries8BitWithSums(
    const Uint8 u8, const Int8 i8, const Int32 i32,
    const uint8_t* HWY_RESTRICT q_data,
    size_t q_stride, size_t num_queries, size_t bytes_per_point,
    size_t num_dim_chunks, size_t N, int8_t* HWY_RESTRICT all_q_decoded,
    int32_t* HWY_RESTRICT all_q_byte_sums,
    const int* indices = nullptr) {
  const size_t decoded_dim = num_dim_chunks * N;
  const auto ones = hn::Set(u8, uint8_t{1});
  for (size_t qi = 0; qi < num_queries; ++qi) {
    const size_t src = indices ? indices[qi] : qi;
    const uint8_t* q_base = q_data + src * q_stride;
    int8_t* q_out = all_q_decoded + qi * decoded_dim;
    hn::Vec<Int32> byte_accum = hn::Zero(i32);
    size_t d = 0;
    for (; d + N <= bytes_per_point; d += N) {
      const auto v = hn::LoadU(i8, reinterpret_cast<const int8_t*>(q_base + d));
      hn::StoreU(v, i8, q_out + d);
      byte_accum =
          hn::SumOfMulQuadAccumulate(i32, ones, v, byte_accum);
    }

    for (size_t k = d; k < decoded_dim; ++k) {
      q_out[k] = (k < bytes_per_point)
                     ? static_cast<int8_t>(q_base[k])
                     : 0;
    }

    int32_t scalar_sum = 0;
    for (size_t k = d; k < bytes_per_point; ++k) {
      scalar_sum += static_cast<int8_t>(q_base[k]);
    }
    all_q_byte_sums[qi] = hn::ReduceSum(i32, byte_accum) + scalar_sum;
  }
}

template <typename Uint8>
HWY_INLINE void DecodeDbPanel8Bit(
    const Uint8 u8,
    const uint8_t* HWY_RESTRICT db_data, size_t db_stride,
    size_t db_start, size_t pi_start, size_t kP, size_t kPoints,
    size_t bytes_per_point, size_t num_dim_chunks, size_t tiles_per_chunk,
    size_t N, uint8_t* HWY_RESTRICT panel,
    const int* indices = nullptr) {
  const size_t panel_bytes = num_dim_chunks * tiles_per_chunk * N;
  if (kP < kPoints) std::memset(panel, 0, panel_bytes);

  using U32 = hn::ScalableTag<uint32_t>;
  using U64 = hn::ScalableTag<uint64_t>;
  const U32 u32;
  const U64 u64;

  const auto bias = hn::Set(u8, uint8_t{128});

  size_t pi = 0;
  for (; pi + 4 <= kP; pi += 4) {
    for (size_t i = 0; i < 4; ++i) {
      if (pi + i + 1 < kP) {
        const size_t next = db_start + pi_start + pi + i + 1;
        absl::PrefetchToLocalCache(db_data + (indices ? indices[next] : next) *
                                                 db_stride);
      }
    }

    for (size_t c = 0; c < num_dim_chunks; ++c) {
      hn::Vec<Uint8> d_vals[4];
      for (size_t i = 0; i < 4; ++i) {
        const size_t cur = db_start + pi_start + pi + i;
        const uint8_t* d_ptr =
            db_data + (indices ? indices[cur] : cur) * db_stride;
        const auto raw = hn::LoadU(u8, d_ptr + c * N);
        d_vals[i] = hn::Xor(raw, bias);
      }

      const size_t co = c * tiles_per_chunk * N;
      auto a = hn::BitCast(u32, d_vals[0]);
      auto b = hn::BitCast(u32, d_vals[1]);
      auto cv = hn::BitCast(u32, d_vals[2]);
      auto d = hn::BitCast(u32, d_vals[3]);

      auto ab_lo = hn::InterleaveLower(u32, a, b);
      auto ab_hi = hn::InterleaveUpper(u32, a, b);
      auto cd_lo = hn::InterleaveLower(u32, cv, d);
      auto cd_hi = hn::InterleaveUpper(u32, cv, d);

      auto t0 =
          hn::BitCast(u32, hn::InterleaveLower(u64, hn::BitCast(u64, ab_lo),
                                               hn::BitCast(u64, cd_lo)));
      auto t1 =
          hn::BitCast(u32, hn::InterleaveUpper(u64, hn::BitCast(u64, ab_lo),
                                               hn::BitCast(u64, cd_lo)));
      auto t2 =
          hn::BitCast(u32, hn::InterleaveLower(u64, hn::BitCast(u64, ab_hi),
                                               hn::BitCast(u64, cd_hi)));
      auto t3 =
          hn::BitCast(u32, hn::InterleaveUpper(u64, hn::BitCast(u64, ab_hi),
                                               hn::BitCast(u64, cd_hi)));

      const auto store_blocks = [&](auto self, auto tag, auto vec, size_t j,
                                    size_t block_base) -> void {
        if constexpr (sizeof(hn::Vec<decltype(tag)>) == 16) {
          const size_t g = j + 4 * block_base;
          const size_t row = g;
          hn::StoreU(
              vec, tag,
              reinterpret_cast<uint32_t*>(panel + co + row * N + pi * 4));
        } else {
          using HalfD = hn::Half<decltype(tag)>;
          const HalfD half_d;
          constexpr size_t kHalfBlocks = sizeof(hn::Vec<HalfD>) / 16;
          self(self, half_d, hn::LowerHalf(half_d, vec), j, block_base);
          self(self, half_d, hn::UpperHalf(half_d, vec), j,
               block_base + kHalfBlocks);
        }
      };
      store_blocks(store_blocks, u32, t0, 0, 0);
      store_blocks(store_blocks, u32, t1, 1, 0);
      store_blocks(store_blocks, u32, t2, 2, 0);
      store_blocks(store_blocks, u32, t3, 3, 0);
    }
  }

  HWY_ALIGN uint8_t tmp[HWY_MAX_BYTES];
  for (; pi < kP; ++pi) {
    if (pi + 1 < kP) {
      const size_t next = db_start + pi_start + pi + 1;
      absl::PrefetchToLocalCache(db_data +
                                 (indices ? indices[next] : next) * db_stride);
    }
    const size_t cur = db_start + pi_start + pi;
    const uint8_t* d_ptr = db_data + (indices ? indices[cur] : cur) * db_stride;
    for (size_t c = 0; c < num_dim_chunks; ++c) {
      const auto raw = hn::LoadU(u8, d_ptr + c * N);
      hn::StoreU(hn::Xor(raw, bias), u8, tmp);
      const size_t co = c * tiles_per_chunk * N;
      const uint32_t* vals32 = reinterpret_cast<const uint32_t*>(tmp);
      for (size_t g = 0; g < N / 4; ++g) {
        reinterpret_cast<uint32_t*>(panel + co + g * N)[pi] = vals32[g];
      }
    }
  }
}

HWY_INLINE uint32_t LoadU32(const uint8_t* p) {
  uint32_t v;
  __builtin_memcpy(&v, p, 4);
  return v;
}

template <typename Uint8>
HWY_INLINE void BuildHammingPanel1Bit(
    const Uint8 u8,
    const uint8_t* HWY_RESTRICT db_data, size_t db_stride,
    size_t db_start, size_t pi_start, size_t kP, size_t kPoints,
    size_t bytes_per_point, size_t num_hamming_tiles, size_t N,
    uint8_t* HWY_RESTRICT panel,
    const int* indices = nullptr) {
  const size_t panel_bytes = num_hamming_tiles * N;
  if (kP < kPoints) std::memset(panel, 0, panel_bytes);

  const size_t full_tiles = bytes_per_point / 4;
  const size_t tail_bytes = bytes_per_point - full_tiles * 4;

  for (size_t pi = 0; pi < kP; ++pi) {
    if (pi + 1 < kP) {
      const size_t next = db_start + pi_start + pi + 1;
      absl::PrefetchToLocalCache(db_data +
                                 (indices ? indices[next] : next) * db_stride);
    }
    const size_t cur = db_start + pi_start + pi;
    const uint8_t* d_ptr = db_data + (indices ? indices[cur] : cur) * db_stride;
    uint32_t* dst = reinterpret_cast<uint32_t*>(panel) + pi;
    const size_t dst_stride = N / 4;

    size_t c = 0;
    for (; c < full_tiles; ++c) {
      dst[c * dst_stride] = LoadU32(d_ptr + c * 4);
    }

    if (tail_bytes > 0) {
      uint32_t w = 0;
      __builtin_memcpy(&w, d_ptr + c * 4, tail_bytes);
      dst[c * dst_stride] = w;
    }
  }
}

static constexpr size_t kMaxSafeAccumTiles = 31;

template <size_t Mq>
HWY_INLINE void PreBroadcastHammingQueries(
    const hn::ScalableTag<uint8_t> u8, const hn::ScalableTag<int32_t> i32,
    const uint8_t* HWY_RESTRICT const* query_packed,
    size_t num_hamming_tiles, size_t bytes_per_point, size_t N,
    uint8_t* HWY_RESTRICT qbuf) {
  const size_t full_tiles = bytes_per_point / 4;
  for (size_t q = 0; q < Mq; ++q) {
    const uint8_t* qp = query_packed[q];
    uint8_t* dst = qbuf + q * num_hamming_tiles * N;
    size_t t = 0;
    for (; t < full_tiles; ++t) {
      hn::Store(hn::BitCast(u8, hn::Set(i32,
          static_cast<int32_t>(LoadU32(qp + t * 4)))), u8, dst + t * N);
    }
    if (t < num_hamming_tiles) {
      uint32_t w = 0;
      const size_t rem = bytes_per_point - full_tiles * 4;
      __builtin_memcpy(&w, qp + t * 4, rem);
      hn::Store(hn::BitCast(u8, hn::Set(i32, static_cast<int32_t>(w))),
                u8, dst + t * N);
    }
  }
}

template <size_t Mq>
HWY_INLINE void HammingAccumLoop(
    const hn::ScalableTag<uint8_t> u8, const hn::ScalableTag<int8_t> i8,
    const hn::ScalableTag<int32_t> i32,
    const uint8_t* HWY_RESTRICT qbuf, size_t qbuf_tile_stride,
    const uint8_t* HWY_RESTRICT panel,
    size_t num_hamming_tiles, size_t N,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc) {
  const auto ones_i8 = hn::Set(i8, int8_t{1});
  for (size_t q = 0; q < Mq; ++q) acc[q] = hn::Zero(i32);

  for (size_t t0 = 0; t0 < num_hamming_tiles; t0 += kMaxSafeAccumTiles) {
    const size_t t_end = std::min(t0 + kMaxSafeAccumTiles, num_hamming_tiles);

    hn::Vec<hn::ScalableTag<uint8_t>> sum[Mq];
    for (size_t q = 0; q < Mq; ++q) sum[q] = hn::Zero(u8);

    for (size_t t = t0; t < t_end; ++t) {
      const auto pv = hn::Load(u8, panel + t * N);
      for (size_t q = 0; q < Mq; ++q) {
        const auto qb = hn::Load(u8, qbuf + q * qbuf_tile_stride + t * N);
        sum[q] = hn::Add(sum[q], hn::PopulationCount(hn::Xor(pv, qb)));
      }
    }

    for (size_t q = 0; q < Mq; ++q) {
      acc[q] = hn::SumOfMulQuadAccumulate(i32, sum[q], ones_i8, acc[q]);
    }
  }
}

template <size_t Mq>
HWY_INLINE void HammingMicroKernelAccumulate1(
    const hn::ScalableTag<uint8_t> u8, const hn::ScalableTag<int8_t> i8,
    const hn::ScalableTag<int32_t> i32,
    const uint8_t* HWY_RESTRICT qbuf, size_t qbuf_tile_stride,
    const uint8_t* HWY_RESTRICT panel,
    size_t num_hamming_tiles, size_t N,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc) {
  HammingAccumLoop<Mq>(u8, i8, i32, qbuf, qbuf_tile_stride,
                        panel, num_hamming_tiles, N, acc);
}

template <size_t Mq>
HWY_INLINE void HammingMicroKernelAccumulate(
    const hn::ScalableTag<uint8_t> u8, const hn::ScalableTag<int8_t> i8,
    const hn::ScalableTag<int32_t> i32,
    const uint8_t* HWY_RESTRICT qbuf, size_t qbuf_tile_stride,
    const uint8_t* HWY_RESTRICT panel0, const uint8_t* HWY_RESTRICT panel1,
    size_t num_hamming_tiles, size_t N,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc0,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc1) {
  const auto ones_i8 = hn::Set(i8, int8_t{1});
  for (size_t q = 0; q < Mq; ++q) {
    acc0[q] = hn::Zero(i32);
    acc1[q] = hn::Zero(i32);
  }

  for (size_t t0 = 0; t0 < num_hamming_tiles; t0 += kMaxSafeAccumTiles) {
    const size_t t_end = std::min(t0 + kMaxSafeAccumTiles, num_hamming_tiles);

    hn::Vec<hn::ScalableTag<uint8_t>> sum0[Mq], sum1[Mq];
    for (size_t q = 0; q < Mq; ++q) {
      sum0[q] = hn::Zero(u8);
      sum1[q] = hn::Zero(u8);
    }

    for (size_t t = t0; t < t_end; ++t) {
      const auto pv0 = hn::Load(u8, panel0 + t * N);
      const auto pv1 = hn::Load(u8, panel1 + t * N);
      for (size_t q = 0; q < Mq; ++q) {
        const auto qb = hn::Load(u8, qbuf + q * qbuf_tile_stride + t * N);
        sum0[q] = hn::Add(sum0[q], hn::PopulationCount(hn::Xor(pv0, qb)));
        sum1[q] = hn::Add(sum1[q], hn::PopulationCount(hn::Xor(pv1, qb)));
      }
    }

    for (size_t q = 0; q < Mq; ++q) {
      acc0[q] = hn::SumOfMulQuadAccumulate(i32, sum0[q], ones_i8, acc0[q]);
      acc1[q] = hn::SumOfMulQuadAccumulate(i32, sum1[q], ones_i8, acc1[q]);
    }
  }
}

template <size_t Mq>
HWY_INLINE void HammingMicroKernelAccumulate4(
    const hn::ScalableTag<uint8_t> u8, const hn::ScalableTag<int8_t> i8,
    const hn::ScalableTag<int32_t> i32,
    const uint8_t* HWY_RESTRICT qbuf, size_t qbuf_tile_stride,
    const uint8_t* HWY_RESTRICT panel0, const uint8_t* HWY_RESTRICT panel1,
    const uint8_t* HWY_RESTRICT panel2, const uint8_t* HWY_RESTRICT panel3,
    size_t num_hamming_tiles, size_t N,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc0,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc1,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc2,
    hn::Vec<hn::ScalableTag<int32_t>>* HWY_RESTRICT acc3) {
  const auto ones_i8 = hn::Set(i8, int8_t{1});
  for (size_t q = 0; q < Mq; ++q) {
    acc0[q] = hn::Zero(i32);
    acc1[q] = hn::Zero(i32);
    acc2[q] = hn::Zero(i32);
    acc3[q] = hn::Zero(i32);
  }

  for (size_t t0 = 0; t0 < num_hamming_tiles; t0 += kMaxSafeAccumTiles) {
    const size_t t_end = std::min(t0 + kMaxSafeAccumTiles, num_hamming_tiles);

    hn::Vec<hn::ScalableTag<uint8_t>> s0[Mq], s1[Mq], s2[Mq], s3[Mq];
    for (size_t q = 0; q < Mq; ++q) {
      s0[q] = hn::Zero(u8);
      s1[q] = hn::Zero(u8);
      s2[q] = hn::Zero(u8);
      s3[q] = hn::Zero(u8);
    }

    for (size_t t = t0; t < t_end; ++t) {
      const auto pa = hn::Load(u8, panel0 + t * N);
      const auto pb = hn::Load(u8, panel1 + t * N);
      const auto pc = hn::Load(u8, panel2 + t * N);
      const auto pd = hn::Load(u8, panel3 + t * N);
      for (size_t q = 0; q < Mq; ++q) {
        const auto qb = hn::Load(u8, qbuf + q * qbuf_tile_stride + t * N);
        s0[q] = hn::Add(s0[q], hn::PopulationCount(hn::Xor(pa, qb)));
        s1[q] = hn::Add(s1[q], hn::PopulationCount(hn::Xor(pb, qb)));
        s2[q] = hn::Add(s2[q], hn::PopulationCount(hn::Xor(pc, qb)));
        s3[q] = hn::Add(s3[q], hn::PopulationCount(hn::Xor(pd, qb)));
      }
    }

    for (size_t q = 0; q < Mq; ++q) {
      acc0[q] = hn::SumOfMulQuadAccumulate(i32, s0[q], ones_i8, acc0[q]);
      acc1[q] = hn::SumOfMulQuadAccumulate(i32, s1[q], ones_i8, acc1[q]);
      acc2[q] = hn::SumOfMulQuadAccumulate(i32, s2[q], ones_i8, acc2[q]);
      acc3[q] = hn::SumOfMulQuadAccumulate(i32, s3[q], ones_i8, acc3[q]);
    }
  }
}

