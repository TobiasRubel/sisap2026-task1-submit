#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace parlayANN {

// Parsed SISAP task-description (config.json). Only the flat scalar fields the
// engine needs are extracted; arrays/objects (e.g. gt_I, queries) are ignored.
struct TaskConfig {
  std::string task;          // e.g. "task1"
  std::string data;          // HDF5 group with the base vectors, e.g. "train"
  long k = 0;                // neighbors to retrieve (15 for task1)
  std::string dataset_name;  // e.g. "wikipedia" / "gooaq-small" (-> dataset attr)
  std::string filename;      // HDF5 file name (used to disambiguate a *.h5 glob)
  bool ok = false;
};

namespace detail {

inline std::string read_whole_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::string();
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Find "key" then the ':' then the next double-quoted string value. Handles
// simple \" and \\ escapes. Not a general JSON parser — sufficient for the
// fixed, flat task config schema.
inline bool get_string(const std::string& s, const std::string& key,
                       std::string& out) {
  const std::string pat = "\"" + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return false;
  p = s.find(':', p + pat.size());
  if (p == std::string::npos) return false;
  size_t q1 = s.find('"', p + 1);
  if (q1 == std::string::npos) return false;
  std::string val;
  size_t i = q1 + 1;
  for (; i < s.size(); ++i) {
    char c = s[i];
    if (c == '\\' && i + 1 < s.size()) {
      val.push_back(s[i + 1]);
      ++i;
      continue;
    }
    if (c == '"') break;
    val.push_back(c);
  }
  if (i >= s.size()) return false;
  out = val;
  return true;
}

// Find "key" then the ':' then a (possibly signed) integer value.
inline bool get_int(const std::string& s, const std::string& key, long& out) {
  const std::string pat = "\"" + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return false;
  p = s.find(':', p + pat.size());
  if (p == std::string::npos) return false;
  ++p;
  while (p < s.size() &&
         (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r'))
    ++p;
  const size_t start = p;
  if (p < s.size() && (s[p] == '+' || s[p] == '-')) ++p;
  const size_t digits = p;
  while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p;
  if (p == digits) return false;
  out = std::atol(s.substr(start, p - start).c_str());
  return true;
}

}  // namespace detail

// Parse config.json; exit(1) on a missing required field (a wrong/empty
// dataset_name would otherwise make the official eval.py silently skip our
// output and score zero, so fail loudly here instead).
inline TaskConfig parse_task_config(const std::string& path) {
  TaskConfig c;
  const std::string s = detail::read_whole_file(path);
  if (s.empty()) {
    std::cerr << "FATAL: cannot read task-description " << path << std::endl;
    std::exit(1);
  }
  detail::get_string(s, "task", c.task);
  detail::get_string(s, "data", c.data);
  detail::get_int(s, "k", c.k);
  detail::get_string(s, "dataset_name", c.dataset_name);
  detail::get_string(s, "filename", c.filename);
  if (c.dataset_name.empty() || c.data.empty() || c.k <= 0) {
    std::cerr << "FATAL: task-description " << path
              << " missing required keys. Parsed: data='" << c.data
              << "' k=" << c.k << " dataset_name='" << c.dataset_name << "'"
              << std::endl;
    std::exit(1);
  }
  c.ok = true;
  return c;
}

}  // namespace parlayANN
