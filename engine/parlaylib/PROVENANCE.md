# Vendored ParlayLib

## Upstream

- Repository: <https://github.com/cmuparlay/parlaylib>
- Commit: `7cdb4cae8f020525f5eb4ad82e2565d1e38cfbc3` ("Elastic scheduler (#52)", 2023-04-04)
- License: MIT (see `LICENSE`)

Only `include/` is vendored (60 headers, ~660 KB). ParlayLib is header-only for our purposes;
its CMake project, tests, benchmarks, and examples are not used.

## Local modification — ALREADY APPLIED

`include/parlay/internal/atomic_wait.h` carries one local hunk, recorded for provenance in
`parlaylib-7cdb4ca-atomic_wait.patch`. **It is already applied to the files in this directory.
Do not run `git apply`.**

The hunk unconditionally `#undef`s `__cpp_lib_atomic_wait` at the top of the header, forcing
ParlayLib to use its own `atomic_wait`/`atomic_notify` implementation everywhere:

```cpp
#ifdef __cpp_lib_atomic_wait
#undef __cpp_lib_atomic_wait
#endif
```

Two problems are avoided:

1. **Clang rejects the standard-library path.** ParlayLib wraps `std::atomic_wait` in a
   variable-template pattern that Clang will not accept as a callable.
2. **ODR violation across translation units.** When TUs are compiled against different C++
   standards, some select the `std::atomic_wait` path and others the custom path. Both define
   the same symbols with different implementations, which is an ODR violation and produces
   subtle link-time and run-time bugs.

Forcing the custom path everywhere resolves both with no measurable performance change.

This patch was compiled into the binary submitted to SISAP 2026 Task 1 — verified by inspecting
the header inside the original build image (`pipnn-build:sub3`). Building without it does not
reproduce the graded artifact.
