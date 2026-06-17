# SISAP 2026 — Task 1 (k-NN graph construction) submission

Binary submission. The container builds **one** quantized index over the input
vectors, then constructs the all-kNN graph under several parameter settings from
that single index, writing one result HDF5 per setting to the output directory.

The source will be open-sourced after the challenge (per arrangement with the
organizers).

## TIRA command

```
/app/neighbors-pipnn -input $inputDataset/*.h5 -task_description $inputDataset/config.json -output $outputDir -data_type float -dist_func mips -turboquant -tq_bits 8 -stream_quantize -build_tq1_panel -query_configs /app/portfolio.task1.txt
```

## Contents

- `neighbors-pipnn` — the compiled engine (reads the HDF5 dataset, builds the
  graph, writes conforming result HDF5s).
- `portfolio.task1.txt` — the parameter settings (one graph build per line).
- `Dockerfile` — runtime image (`ubuntu:22.04` + the shared libraries the binary needs).
