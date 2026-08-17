#pragma once

#include <hdf5.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace parlayANN {

// Write a SISAP 2026 result HDF5 conforming to the official baseline's
// store_results / eval.py:
//   datasets: knns (int32), dists (float32), both shape n x ncols
//   attrs:    algo, dataset, task, params  (variable-length UTF-8 strings,
//             so h5py/eval.py read them back as Python str, not bytes)
//             buildtime, querytime         (float64 scalars)
// `knns`/`dists` are caller-owned, row-major n x ncols, already 1-based with
// the point's own id at column 0 (the all-kNN self-match the grader expects).
inline void write_sisap_result(const std::string& path, const int32_t* knns,
                               const float* dists, size_t n, long ncols,
                               const std::string& algo,
                               const std::string& dataset,
                               const std::string& task,
                               const std::string& params, double buildtime,
                               double querytime) {
  hid_t fid = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (fid < 0) {
    std::cerr << "FATAL: H5Fcreate failed for " << path << std::endl;
    std::abort();
  }

  hsize_t dims[2] = {static_cast<hsize_t>(n), static_cast<hsize_t>(ncols)};
  hid_t sp = H5Screate_simple(2, dims, nullptr);

  hid_t dk = H5Dcreate2(fid, "knns", H5T_STD_I32LE, sp, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dk, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, knns);
  H5Dclose(dk);

  hid_t dd = H5Dcreate2(fid, "dists", H5T_IEEE_F32LE, sp, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dd, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dists);
  H5Dclose(dd);
  H5Sclose(sp);

  hid_t asp = H5Screate(H5S_SCALAR);

  hid_t st = H5Tcopy(H5T_C_S1);
  H5Tset_size(st, H5T_VARIABLE);
  H5Tset_cset(st, H5T_CSET_UTF8);
  auto write_str = [&](const char* name, const std::string& v) {
    hid_t at = H5Acreate2(fid, name, st, asp, H5P_DEFAULT, H5P_DEFAULT);
    const char* p = v.c_str();
    H5Awrite(at, st, &p);
    H5Aclose(at);
  };
  write_str("algo", algo);
  write_str("dataset", dataset);
  write_str("task", task);
  write_str("params", params);
  H5Tclose(st);

  auto write_f64 = [&](const char* name, double v) {
    hid_t at = H5Acreate2(fid, name, H5T_IEEE_F64LE, asp, H5P_DEFAULT,
                          H5P_DEFAULT);
    H5Awrite(at, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(at);
  };
  write_f64("buildtime", buildtime);
  write_f64("querytime", querytime);

  H5Sclose(asp);
  H5Fclose(fid);

  std::cout << "Wrote SISAP result " << path << " (" << n << " x " << ncols
            << ", dataset=" << dataset << ", buildtime=" << buildtime
            << "s querytime=" << querytime << "s)" << std::endl;
}

}  // namespace parlayANN
