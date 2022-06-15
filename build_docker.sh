#!/bin/bash

mkdir target

wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz -O target/onnxruntime.tgz

tar zxvf target/onnxruntime.tgz -C target

cargo build --release

docker build -t lz1998/nsfw . --no-cache