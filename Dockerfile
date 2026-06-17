FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# Runtime shared libraries the prebuilt binary links against (from `readelf -d`):
# libhdf5_serial.so.103, libcrypto.so.3 (libssl3), libcurl.so.4, libsz.so.2,
# libz.so.1, libomp.so.5, libatomic.so.1. libstdc++6/libgcc-s1/libc6/libm are in
# the base image. libhdf5 also pulls curl/ssl/szip transitively.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libhdf5-103-1 \
      libcurl4 \
      libssl3 \
      libsz2 \
      zlib1g \
      libomp5-14 \
      libatomic1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY neighbors-pipnn /app/neighbors-pipnn
COPY portfolio.task1.txt /app/portfolio.task1.txt

# No ENTRYPOINT/CMD: TIRA supplies the full command.
