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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../bench/parse_command_line.h"
#include "../utils/graph.h"
#include "../utils/types.h"
#include "pipnn.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

namespace parlayANN {

template <typename Point, typename PointRange, typename indexType>
void ANN(long k, BuildParams &BP, PointRange &Points, commandLine &P) {
  using findex = pipnn_index<Point, PointRange, indexType>;

  char* qconfigs = P.getOptionValue("-query_configs");
  if (!qconfigs) {
    std::cerr << "FATAL: -query_configs <file> is required for this code..."
              << std::endl;
    std::abort();
  }

  // Multi-query mode: encode the RSQ index ONCE, then build N kNN
  // graphs from it (one per non-blank line in the -query_configs file). The
  // encoded codes stay resident in RAM (cached on the index object `I`) and are
  // never serialized — this is the SISAP "one index build, many query builds"
  // model. Each config line carries its own build-param overrides; the line's
  // tokens are prepended to argv so they win commandLine::getOptionValue
  // (which returns the first match).
  {
    std::ifstream cf(qconfigs);
    if (!cf) {
      std::cerr << "FATAL: cannot open -query_configs " << qconfigs << std::endl;
      std::abort();
    }
    findex I;  // holds the cached encoded index across all builds
    std::string line;
    int cfg_idx = 0;
    double shared_buildtime = 0.0;
    bool have_buildtime = false;
    while (std::getline(cf, line)) {
      size_t first = line.find_first_not_of(" \t\r\n");
      if (first == std::string::npos || line[first] == '#') continue;
      std::vector<std::string> toks;
      {
        std::stringstream ss(line);
        std::string tk;
        while (ss >> tk) toks.push_back(tk);
      }
      if (toks.empty()) continue;
      // Per-config argv = [prog] + config-override tokens + base argv[1..].
      std::vector<char*> av;
      av.push_back(P.argv[0]);
      for (auto& tk : toks) av.push_back(const_cast<char*>(tk.c_str()));
      for (int i = 1; i < P.argc; ++i) av.push_back(P.argv[i]);
      // A portfolio line may itself use a legacy spelling (e.g. -build_tq1_panel),
      // and these tokens are read from Pc, not from main's P — so canonicalize
      // here too. Getting this wrong makes the flag a silent no-op.
      canonicalize_flags(av);
      commandLine Pc(static_cast<int>(av.size()), av.data());

      // Re-derive build_topk's explicit args from the per-config commandLine
      // (config overrides win because getOptionValue returns the first match).
      // (-fanout and -alpha also appear in every portfolio line and are
      // deliberately ignored; effective fanout comes exclusively from
      // -fanout_scheme.)
      long c_rounds = Pc.getOptionIntValue("-num_clusters", 1);
      long c_size = Pc.getOptionIntValue("-cluster_size", 512);
      long c_mst = Pc.getOptionIntValue("-mst_deg", 2);
      long c_tll = Pc.getOptionLongValue("-top_level_leaders", 1024);
      double c_frl = Pc.getOptionDoubleValue("-fraction_leaders", 0.005);
      std::string c_fs = Pc.getOptionValue("-fanout_scheme", "10,3,1");

      Graph<indexType> Gc;  // fresh per config; freed at end of iteration
      double enc_s = 0.0, build_s = 0.0;
      // One shared index, many query builds: announce this query. The encode
      // happens on the first query only (build_topk caches + reuses the codes);
      // every later query reuses that single index (it logs "encode skipped").
      const char* cfg_name = Pc.getOptionValue("-cfg_name");
      const char* cfg_params = Pc.getOptionValue("-cfg_params");
      std::cout << "\n=== Query " << (cfg_idx + 1) << ": building kNN graph from "
                << (cfg_idx == 0 ? "the shared index (this first query also "
                                   "reads + quantizes the vectors once)"
                                 : "the shared index (reusing cached codes)")
                << "\nQuerying with " << (cfg_name ? cfg_name : "config") << ": "
                << (cfg_params ? cfg_params : "(default params)") << std::endl;
      I.build_topk(Gc, Pc, Points, c_rounds, c_size, c_mst, k, c_tll,
                   c_frl, c_fs, BP.rsq, BP.rsq_bits, &enc_s, &build_s,
                   /*buildtime_in=*/shared_buildtime);
      if (!have_buildtime) {  // only the first build encodes
        shared_buildtime = enc_s;
        have_buildtime = true;
      }
      // Machine-parseable result line (log-only; the HDF5 written via -output
      // is the actual artifact).
      std::cout << "MULTIQ_RESULT idx=" << cfg_idx
                << " buildtime=" << shared_buildtime
                << " querytime=" << build_s << std::endl;
      cfg_idx++;
    }
    std::cout << "MULTIQ_BUILDTIME " << shared_buildtime << std::endl;
    std::cout << "MULTIQ_DONE configs=" << cfg_idx << std::endl;
    return;
  }
}

};  // namespace parlayANN
