// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
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

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "parlay/parallel.h"

#include <hdf5.h>

namespace parlayANN {

template<class Point_>
struct PointRange{
  using Point = Point_;
  using parameters = typename Point::parameters;
  using byte = uint8_t;

  long dimension() const {return params.dims;}

  // HDF5 reader — the only load path in this build. `dataset` names a 2-D
  // array of shape (n, d), element type f32 or f16.
  //
  // Two live layouts:
  //  - chunked/filtered (e.g. gzip; the SISAP dry-run corpus): read in full
  //    through H5Dread into the aligned float arena; the non-streaming encode
  //    path in build_topk consumes it.
  //  - contiguous (the production wikipedia corpus): metadata only, arena
  //    deferred; init_rsq_streaming re-mmaps the file and encodes
  //    chunk-by-chunk. Requires -stream_quantize (always set by the
  //    submission command); this build aborts otherwise rather than carrying
  //    the eager mmap+decode path the submission never ran.
  PointRange(char* h5_path, char* dataset, bool defer_load)
      : values(std::shared_ptr<byte[]>(nullptr, std::free)) {
    // --- 1. Pull metadata from HDF5: dataset offset, shape, element size/class.
    hid_t fid = H5Fopen(h5_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) {
      std::cout << "H5Fopen failed: " << h5_path << std::endl;
      std::abort();
    }
    hid_t did = H5Dopen2(fid, dataset, H5P_DEFAULT);
    if (did < 0) {
      std::cout << "H5Dopen failed for dataset '" << dataset << "' in "
                << h5_path << std::endl;
      std::abort();
    }
    haddr_t h5_offset = H5Dget_offset(did);
    // A non-contiguous (chunked/compact/virtual or filtered/gzip) dataset has
    // no single file offset, so the mmap+offset streaming path cannot be used.
    // Such datasets are read in full via H5Dread instead (see the fallback
    // after the metadata block). The SISAP dry-run corpus (gooaq-small) is
    // chunked+gzip; the production wikipedia corpus is contiguous.
    const bool is_chunked = (h5_offset == HADDR_UNDEF);
    hid_t sid = H5Dget_space(did);
    int ndims = H5Sget_simple_extent_ndims(sid);
    if (ndims != 2) {
      std::cout << "Dataset '" << dataset << "' has ndims=" << ndims
                << "; expected 2-D (n, d)." << std::endl;
      std::abort();
    }
    hsize_t dims[2];
    H5Sget_simple_extent_dims(sid, dims, nullptr);
    hid_t tid = H5Dget_type(did);
    size_t src_elem_bytes = H5Tget_size(tid);
    H5T_class_t tclass = H5Tget_class(tid);
    bool src_is_float = (tclass == H5T_FLOAT);
    H5Tclose(tid);
    H5Sclose(sid);
    H5Dclose(did);
    H5Fclose(fid);

    n = dims[0];
    unsigned int d = static_cast<unsigned int>(dims[1]);
    params = parameters(d);
    int num_bytes = params.num_bytes();
    aligned_bytes = (num_bytes <= 32) ? 32 : 64 * ((num_bytes - 1) / 64 + 1);

    std::cout << "HDF5: " << h5_path << ":" << dataset << " n=" << n
              << " d=" << d << " src_elem_bytes=" << src_elem_bytes
              << " dst_bytes_per_point=" << num_bytes
              << " offset=" << h5_offset
              << (is_chunked ? " (chunked/filtered: H5Dread fallback)" : "")
              << std::endl;

    if (is_chunked) {
      // Chunked/filtered layout: read the whole dataset through the HDF5
      // library (it decompresses gzip and converts f16->f32 for us) into a
      // temporary float buffer, then pack into the aligned arena. Streaming is
      // not possible here, so we materialize and disable deferred load — the
      // non-streaming encode path in build_topk consumes this arena. (Used by
      // the SISAP dry-run corpus; the production corpus is contiguous.)
      if (!src_is_float || (size_t)num_bytes != (size_t)d * sizeof(float)) {
        std::cout << "Unsupported chunked HDF5 dtype: src_is_float="
                  << src_is_float << " src_elem_bytes=" << src_elem_bytes
                  << " dst_bytes_per_point=" << num_bytes
                  << " (expected float points, d*4 bytes)." << std::endl;
        std::abort();
      }
      long total_bytes = n * aligned_bytes;
      constexpr size_t ALIGNMENT = 1L << 21;
      total_bytes = (total_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
      byte* ptr = (byte*)aligned_alloc(ALIGNMENT, total_bytes);
      madvise(ptr, total_bytes, MADV_HUGEPAGE);
      values = std::shared_ptr<byte[]>(ptr, std::free);
      byte* values_ptr = values.get();

      // Re-open the dataset (the handles above were already closed) and read
      // all rows as native float; HDF5 handles the gzip filter + f16->f32.
      hid_t rfid = H5Fopen(h5_path, H5F_ACC_RDONLY, H5P_DEFAULT);
      hid_t rdid = H5Dopen2(rfid, dataset, H5P_DEFAULT);
      std::vector<float> fbuf((size_t)n * d);
      if (H5Dread(rdid, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  fbuf.data()) < 0) {
        std::cout << "H5Dread failed for chunked dataset '" << dataset << "'"
                  << std::endl;
        std::abort();
      }
      H5Dclose(rdid);
      H5Fclose(rfid);

      parlay::parallel_for(0, n, [&](size_t i) {
        std::memcpy(values_ptr + i * aligned_bytes, fbuf.data() + i * d,
                    (size_t)num_bytes);
      });
      deferred_load = false;
      return;
    }

    if (!defer_load) {
      std::cout << "FATAL: contiguous HDF5 datasets require -stream_quantize "
                   "in this build." << std::endl;
      std::abort();
    }

    // Skip the 25 GB anon allocation and parallel memcpy. Caller (e.g.
    // init_rsq_streaming) will re-mmap the file using these fields
    // and encode chunk-by-chunk with MADV_DONTNEED, so the float arena is
    // never fully resident. Used by Task 1 with -stream_quantize.
    struct stat sbf;
    if (stat(h5_path, &sbf) != 0) {
      std::cout << "stat() failed: " << h5_path << std::endl;
      std::abort();
    }
    deferred_path = h5_path;
    deferred_h5_offset = h5_offset;
    deferred_src_elem_bytes = src_elem_bytes;
    deferred_src_is_float = src_is_float;
    deferred_file_size = static_cast<size_t>(sbf.st_size);
    deferred_load = true;
    std::cout << "PointRange: deferred-load mode (no anon arena allocated; "
              << "file_size=" << deferred_file_size << ")" << std::endl;
  }

  static PointRange from_path(char* path, char* dataset, bool defer_load) {
    if (path == nullptr || dataset == nullptr) {
      std::cout << "FATAL: from_path requires an HDF5 path and dataset name."
                << std::endl;
      std::abort();
    }
    return PointRange(path, dataset, defer_load);
  }

  size_t size() const { return n; }

  Point operator [] (long i) const {
    if (i > n) {
      std::cout << "ERROR: point index out of range: " << i << " from range " << n << ", " << std::endl;
      abort();
    }
    return Point(values.get()+i*aligned_bytes, i, params);
  }

  parameters params;

  std::shared_ptr<byte[]> values;
  long aligned_bytes;
  size_t n;

  // Deferred-load mode: when true, the HDF5 constructor parsed metadata
  // (n, d, aligned_bytes, params, plus the fields below) but skipped the
  // anon-arena allocation and parallel memcpy. `values` is null in this
  // mode; callers that can stream from the file (e.g. init_rsq_streaming)
  // re-mmap using `deferred_path`/`deferred_h5_offset` and consume chunks.
  // Touching `values.get()` or operator[] in this mode is a programming error.
  bool deferred_load = false;
  std::string deferred_path;
  size_t deferred_h5_offset = 0;
  size_t deferred_src_elem_bytes = 0;
  bool deferred_src_is_float = false;
  size_t deferred_file_size = 0;
};

} // end namespace
