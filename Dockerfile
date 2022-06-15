FROM debian:buster-slim

COPY target/release/libnsfw /libnsfw/libnsfw
COPY target/onnxruntime-linux-x64-1.8.1/lib/libonnxruntime.so.1.8.1 /libnsfw/libonnxruntime.so.1.8.1
COPY nsfw.onnx /libnsfw/nsfw.onnx

CMD LD_LIBRARY_PATH=/libnsfw /libnsfw/libnsfw  0.0.0.0:8000 /libnsfw/nsfw.onnx 4
