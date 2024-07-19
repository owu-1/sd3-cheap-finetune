FROM debian:bookworm-20240701 as build

RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    clang-16 \
    git \
    libbrotli-dev \
    libgif-dev \
    libjpeg-dev \
    libopenexr-dev \
    libpng-dev \
    libwebp-dev \
    pkg-config

ENV LIBJXL_VERSION v0.10.3

# Gives error if --shallow-submodules is passed to git clone
RUN git clone https://github.com/libjxl/libjxl.git && \
    cd libjxl && \
    git checkout $LIBJXL_VERSION && \
    git submodule update --init --recursive --depth 1 --recommend-shallow

ENV CC clang-16
ENV CXX clang++-16

WORKDIR /libjxl

RUN mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF .. && \
    cmake --build . -- -j$(nproc)


FROM debian:bookworm-20240701

COPY --from=build /libjxl/build /build

RUN apt-get update && \
    apt-get install -y \
    libbrotli-dev \
    libgif-dev \
    libjpeg-dev \
    libopenexr-dev \
    libpng-dev \
    libwebp-dev

# Use bash shell for brace expansion
SHELL ["/bin/bash", "-c"]

RUN mv /build/tools/{benchmark_xl,cjpegli,cjxl,djpegli,djxl,jxlinfo} /usr/local/bin/ && \
    mv /build/lib/libjxl*.so* /usr/local/lib && \
    rm /build -r && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/libjxl.conf && \
    ldconfig

# Reset to default shell
SHELL ["/bin/sh", "-c"]
