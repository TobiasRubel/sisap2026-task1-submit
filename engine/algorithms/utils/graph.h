// This code is part of the Parlay Project
// Copyright (c) 2024 Guy Blelloch, Magdalen Dobson and the Parlay team
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
#include <iostream>
#include <limits>
#include <sys/mman.h>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/alloc.h"

#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

namespace parlayANN {
template<typename indexType>
struct edgeRange {
  edgeRange(indexType* start, indexType* end, indexType id, absl::Mutex* lock)
    : edges(parlay::make_slice<indexType*, indexType*>(start,end)), id_(id), lock(lock) {
    maxDeg = edges.size() - 1;
  }

  indexType* begin() const {return edges.begin() + 1;}

private:
  parlay::slice<indexType*, indexType*> edges;
  long maxDeg;
  indexType id_;
  absl::Mutex* lock;
};

template<typename indexType_>
struct Graph {
  using indexType = indexType_;

  Graph(){}

  void allocate_graph(long _maxDeg, size_t _n) {
    maxDeg = _maxDeg;
    n = _n;
    long cnt = n * (maxDeg + 1);
    long num_bytes = cnt * sizeof(indexType);
    indexType* ptr;
    if (num_bytes > 1 << 24) {
      constexpr size_t ALIGNMENT = 1L << 21;
      num_bytes = (num_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
      ptr = (indexType*) aligned_alloc(ALIGNMENT, num_bytes);
      madvise(ptr, num_bytes, MADV_HUGEPAGE);
      graph = std::shared_ptr<indexType[]>(ptr, std::free);
    }
    else {
      ptr = (indexType*) parlay::p_malloc(num_bytes);
      graph = std::shared_ptr<indexType[]>(ptr, parlay::p_free);
    }
    parlay::parallel_for(0, cnt, [&] (long i) {ptr[i] = 0;});
    locks = new absl::Mutex[n];
  }

  edgeRange<indexType> operator [] (indexType i) {
    if (i > n) {
      std::cout << "ERROR: graph index out of range: " << i << std::endl;
      abort();
    }
    return edgeRange<indexType>(graph.get() + i * (maxDeg + 1),
                                graph.get() + (i + 1) * (maxDeg + 1),
                                i, locks + i);
  }

  ~Graph(){}

private:
  size_t n;
  long maxDeg;
  std::shared_ptr<indexType[]> graph;
  absl::Mutex* locks;
};

template<typename indexType_>
struct NghDistGraphHeap {
public:
  using indexType = indexType_;
  using T = std::pair<indexType, float>;

  size_t size() const {return n;}

  static constexpr T kEmptyCell = std::make_pair(0, std::numeric_limits<float>::max());

  NghDistGraphHeap(){}

  void set_k_max(size_t k) { kMaxDegree = k; }

  void allocate_graph(size_t n_, size_t k_max) {
    n = n_;
    kMaxDegree = k_max;
    if (locks) { delete[] locks; locks = nullptr; }
    if (tables) { std::free(tables); tables = nullptr; }
    locks = new absl::Mutex[n];
    constexpr size_t ALIGNMENT = 1L << 21;
    size_t num_bytes = ((kMaxDegree+1)*n*sizeof(T));
    num_bytes = (num_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    tables = (T*) aligned_alloc(ALIGNMENT, num_bytes);
    madvise(tables, num_bytes, MADV_HUGEPAGE);
    std::cout << "NghDistGraphHeap: " << n << " points, k_max=" << kMaxDegree
              << ", allocated " << num_bytes / (1024.0 * 1024.0) << " MiB"
              << std::endl;
    parlay::internal::timer t;
    t.start();
    parlay::parallel_for(0, (kMaxDegree+1)*n, [&] (size_t i) {
      tables[i] = kEmptyCell;
    });
    t.next("Table initialization time");
  }

  void allocate_graph(size_t n_) { allocate_graph(n_, kMaxDegree); }

  size_t degree(size_t i) const {
    return tables[i*(kMaxDegree+1)].first;
  }

  absl::Span<T> Neighbors(size_t i) {
    size_t offset = i * (kMaxDegree + 1);
    size_t d = degree(i);
    return absl::Span<T>(tables + offset + 1, d);
  }

  static bool dist_comparator(const T& a, const T& b) {
    return a.second < b.second;
  }

  template<typename rangeType>
  void merge_neighbors(size_t i, const rangeType& r) {
    locks[i].Lock();
    size_t offset = i * (kMaxDegree + 1);
    T* neighbors = tables + offset + 1;
    uint32_t& current_degree = tables[offset].first;

    auto is_present = [&] (const auto& ngh) {
      for (size_t j = 0; j < current_degree; ++j) {
        if (ngh.first == neighbors[j].first) return true;
      }
      return false;
    };

    for (const auto& ngh : r) {
      if (current_degree == kMaxDegree &&
          ngh.second >= neighbors[0].second) {
        break;
      }
      if (is_present(ngh)) continue;
      if (current_degree < kMaxDegree) {
        neighbors[current_degree] = ngh;
        current_degree++;
        std::push_heap(neighbors, neighbors + current_degree, dist_comparator);
      } else {
        std::pop_heap(neighbors, neighbors + kMaxDegree, dist_comparator);
        neighbors[kMaxDegree - 1] = ngh;
        std::push_heap(neighbors, neighbors + kMaxDegree, dist_comparator);
      }
    }

    locks[i].Unlock();
  }

  void sort_heap(size_t i) {
    size_t offset = i * (kMaxDegree + 1);
    T* neighbors = tables + offset + 1;
    size_t current_degree = tables[offset].first;
    std::sort_heap(neighbors, neighbors + current_degree, dist_comparator);
  }

  void finalize() {
    parlay::parallel_for(0, n, [&] (size_t i) {
      sort_heap(i);
    });
  }

  ~NghDistGraphHeap() {
    if (locks) delete[] locks;
    if (tables) std::free(tables);
  }

  size_t n = 0;
  size_t kMaxDegree = 1024;
  absl::Mutex* locks = nullptr;
  T* tables = nullptr;
};

}

