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

#include <iostream>
#include <string>
#include <vector>

#include "parse_command_line.h"
#include "time_loop.h"
#include "../utils/point_range.h"
#include "../utils/mips_point.h"
#include "../utils/task_config.h"
#include "../utils/types.h"

using namespace parlayANN;

using uint = unsigned int;

// ANN<>() arrives via the precompiled neighbors.h (see CMakeLists.txt); this
// file deliberately does not #include it.
template<typename Point, typename PointRange, typename indexType>
void timeNeighbors(long k, BuildParams &BP, PointRange &Points, commandLine& P)
{
    time_loop(1, 0,
      [&] () {},
      [&] () {
        ANN<Point, PointRange, indexType>(k, BP, Points, P);
      },
      [&] () {});
}

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
      "-input <dataset.h5> -task_description <config.json> -output <dir> "
      "-data_type float -dist_func mips -rsq [-rsq_bits <8|1>] "
      "[-stream_quantize] [-build_rsq1_panel] -query_configs <file>");

  // Accept the pre-rename flag spellings (see canonical_flag). This must happen
  // before the first getOption* call, and canon_argv must outlive every use of
  // P — hence main scope.
  std::vector<char*> canon_argv(argv, argv + argc);
  if (canonicalize_flags(canon_argv)) {
    std::cout << "note: accepted legacy flag spellings; the current names are "
                 "-rsq / -rsq_bits / -build_rsq1_panel"
              << std::endl;
  }
  P.argv = canon_argv.data();
  P.argc = static_cast<int>(canon_argv.size());

  char* taskDesc = P.getOptionValue("-task_description");
  char* inputH5 = P.getOptionValue("-input");
  if (taskDesc == nullptr || inputH5 == nullptr) {
    std::cerr << "FATAL: -input <dataset.h5> and -task_description <config.json>"
                 " are both required. This build only supports the SISAP 2026"
                 " Task 1 submission invocation." << std::endl;
    return 1;
  }
  // These backing strings + the augmented argv must outlive every use of P
  // (timeNeighbors/ANN run later in main), so they are declared at main scope.
  std::string sub_input, sub_dset_group, sub_dset_name, sub_task_name;
  std::string dn_key = "-dataset_name", tn_key = "-task_name";
  std::vector<char*> sub_argv;
  TaskConfig cfg = parse_task_config(taskDesc);  // exits on missing keys
  // Resolve the dataset file: prefer the -input value; if its basename does
  // not match config.filename (e.g. the *.h5 glob expanded to several files,
  // leaving extras as stray argv tokens), scan argv for a matching token.
  auto ends_with = [](const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
  };
  sub_input = inputH5;
  if (!cfg.filename.empty() && !ends_with(sub_input, cfg.filename)) {
    for (int i = 1; i < argc; ++i)
      if (ends_with(std::string(argv[i]), cfg.filename)) { sub_input = argv[i]; break; }
  }
  sub_dset_group = cfg.data;
  sub_dset_name = cfg.dataset_name;
  sub_task_name = cfg.task.empty() ? std::string("task1") : cfg.task;
  char* iFile = const_cast<char*>(sub_input.c_str());
  char* iDset = const_cast<char*>(sub_dset_group.c_str());
  long k = cfg.k;
  if (k <= 0 || k > 1000) {
    std::cerr << "FATAL: config.json k=" << k << " out of range (1..1000)"
              << std::endl;
    return 1;
  }
  // Augment argv so the -query_configs loop's per-config commandLine exposes
  // -dataset_name / -task_name to build_topk's writer.
  // From P.argv, not the raw argv: the raw one still holds legacy spellings.
  sub_argv.assign(P.argv, P.argv + P.argc);
  sub_argv.push_back(const_cast<char*>(dn_key.c_str()));
  sub_argv.push_back(const_cast<char*>(sub_dset_name.c_str()));
  sub_argv.push_back(const_cast<char*>(tn_key.c_str()));
  sub_argv.push_back(const_cast<char*>(sub_task_name.c_str()));
  P.argv = sub_argv.data();
  P.argc = static_cast<int>(sub_argv.size());
  std::cout << "SISAP submission mode: input=" << iFile
            << " dataset_group=" << iDset << " k=" << k
            << " dataset_name=" << sub_dset_name << " task=" << sub_task_name
            << std::endl;

  char* vectype = P.getOptionValue("-data_type");
  char* dfc = P.getOptionValue("-dist_func");
  if (vectype == nullptr || std::string(vectype) != "float") {
    std::cerr << "FATAL: -data_type must be 'float'. This build ships only the "
                 "FLOAT_T x MIPS variant used by the SISAP 2026 Task 1 "
                 "submission." << std::endl;
    return 1;
  }
  if (dfc == nullptr || std::string(dfc) != "mips") {
    std::cerr << "FATAL: -dist_func must be 'mips'. This build ships only the "
                 "FLOAT_T x MIPS variant used by the SISAP 2026 Task 1 "
                 "submission." << std::endl;
    return 1;
  }

  BuildParams BP;
  BP.rsq = P.getOption("-rsq");
  if (!BP.rsq) {
    std::cerr << "FATAL: -rsq (legacy: -turboquant) is required; this build "
                 "scores exclusively on the quantized codes." << std::endl;
    return 1;
  }
  BP.rsq_bits = P.getOptionIntValue("-rsq_bits", 8);
  if (BP.rsq_bits != 8 && BP.rsq_bits != 1) {
    std::cerr << "FATAL: -rsq_bits must be 8 or 1, got " << BP.rsq_bits
              << ". This build supports only the 8-bit codes (RSQ8) and the "
                 "1-bit sign panel (RSQ1)." << std::endl;
    return 1;
  }
  std::cout << "RSQ quantized scoring enabled (" << BP.rsq_bits << "-bit)" << std::endl;

  // -stream_quantize: defer the float arena allocation in the HDF5 PointRange
  // load. The build path (init_rsq_streaming) re-mmaps the file and
  // encodes chunk-by-chunk to avoid materializing the 25 GB float arena.
  // Chunked/gzip datasets are materialized through H5Dread either way.
  bool stream_quantize = P.getOption("-stream_quantize");

  using Point = Mips_Point<float>;
  using PR = PointRange<Point>;
  PR Points = PR::from_path(iFile, iDset, stream_quantize);
  timeNeighbors<Point, PR, uint>(k, BP, Points, P);

  return 0;
}
