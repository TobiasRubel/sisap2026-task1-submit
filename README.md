# PiPNN — SISAP 2026 Indexing Challenge, Task 1

Our entry for Task 1 (all-kNN graph construction), and the source that produces it.

The engine builds **one** quantized index over the input vectors, then constructs the all-kNN
graph under 14 parameter settings from that single index, writing one result HDF5 per setting.

| path | what it is |
|---|---|
| `neighbors-pipnn` | the binary that was graded, exactly as submitted |
| `portfolio.task1.txt` | the 14 parameter settings |
| `Dockerfile` | the runtime image TIRA built and ran |
| `engine/` | source for the graph builder |
| `Dockerfile.build` | builds `engine/` and produces the same runtime image |
| `verify/` | scripts to run a binary on a corpus and compare two result sets |

## Run

```
/app/neighbors-pipnn -input $inputDataset/*.h5 -task_description $inputDataset/config.json \
  -output $outputDir -data_type float -dist_func mips -rsq -rsq_bits 8 \
  -stream_quantize -build_rsq1_panel -query_configs /app/portfolio.task1.txt
```

The graded run used `-turboquant -tq_bits 8 -build_tq1_panel`, which the engine still accepts as
aliases for the three flags above.

## Build

```
docker build -f Dockerfile.build -t pipnn:rebuilt .
```

Needs network at configure time: CMake fetches Eigen, Abseil, Highway and mimalloc, each pinned
to the commit used for the submission.

## Scope

This engine builds all-kNN graphs with 8-bit rotated scalar quantization (RSQ8) and the derived
1-bit sign panel (RSQ1), for the parameter ranges the submission used. Outside them it **aborts**
rather than silently falling back — the `-input`/`-task_description` invocation above is the only
input mode, `-rsq` and `-query_configs` are required, `-rsq_bits` accepts only 8 or 1, `-leaf_k`
only 3–15, and contiguous datasets require `-stream_quantize`.

MIT licensed — see `LICENSE`; third-party components in `NOTICE`.
