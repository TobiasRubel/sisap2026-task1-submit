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

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Legacy flag spellings.
//
// The command submitted to SISAP 2026 Task 1 used the spellings on the left,
// and that command is a historical record which must keep working verbatim.
// So argv is canonicalized
// once, up front, and this is the ONLY place the old names are known — every
// getOption* call downstream sees the modern spelling.
//
// Canonicalizing argv (rather than teaching the getters about aliases) also
// preserves the first-match-wins behaviour that per-config portfolio overrides
// depend on: a config line's tokens are prepended to argv, so whichever
// spelling the line uses still wins over the base command line.
// ---------------------------------------------------------------------------
inline const char* canonical_flag(const char* tok) {
  static const std::pair<const char*, const char*> kAliases[] = {
      {"-turboquant",      "-rsq"},
      {"-tq_bits",         "-rsq_bits"},
      {"-build_tq1_panel", "-build_rsq1_panel"},
  };
  for (const auto& [legacy, modern] : kAliases)
    if (std::strcmp(tok, legacy) == 0) return modern;
  return tok;
}

// Rewrites legacy flag spellings in place. Returns true if any were found.
// The replacements are string literals (static storage), so they outlive argv.
inline bool canonicalize_flags(std::vector<char*>& av) {
  bool saw_legacy = false;
  for (size_t i = 1; i < av.size(); ++i) {
    const char* c = canonical_flag(av[i]);
    if (c != av[i]) { saw_legacy = true; av[i] = const_cast<char*>(c); }
  }
  return saw_legacy;
}

struct commandLine {
  int argc;
  char** argv;
  std::string comLine;
  commandLine(int _c, char** _v, std::string _cl)
    : argc(_c), argv(_v), comLine(_cl) {
      if (getOption("-h") || getOption("-help"))
	badArgument();
    }

  commandLine(int _c, char** _v)
    : argc(_c), argv(_v), comLine("bad arguments") { }

  void badArgument() {
    std::cout << "usage: " << argv[0] << " " << comLine << std::endl;
    exit(0);
  }

  bool getOption(std::string option) {
    for (int i = 1; i < argc; i++)
      if ((std::string) argv[i] == option) return true;
    return false;
  }

  char* getOptionValue(std::string option) {
    for (int i = 1; i < argc-1; i++)
      if ((std::string) argv[i] == option) return argv[i+1];
    return NULL;
  }

  std::string getOptionValue(std::string option, std::string defaultValue) {
    for (int i = 1; i < argc-1; i++)
      if ((std::string) argv[i] == option) return (std::string) argv[i+1];
    return defaultValue;
  }

  long getOptionLongValue(std::string option, long defaultValue) {
    for (int i = 1; i < argc-1; i++)
      if ((std::string) argv[i] == option) {
	long r = atol(argv[i+1]);
	if (r < 0) badArgument();
	return r;
      }
    return defaultValue;
  }

  int getOptionIntValue(std::string option, int defaultValue) {
    for (int i = 1; i < argc-1; i++)
      if ((std::string) argv[i] == option) {
	int r = atoi(argv[i+1]);
	if (r < 0) badArgument();
	return r;
      }
    return defaultValue;
  }

  double getOptionDoubleValue(std::string option, double defaultValue) {
    for (int i = 1; i < argc-1; i++)
      if ((std::string) argv[i] == option) {
	double val;
	if (sscanf(argv[i+1], "%lf",  &val) == EOF) {
	  badArgument();
	}
	return val;
      }
    return defaultValue;
  }

};

