## download onnxruntime pre-compiled version:
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.9.0/onnxruntime-linux-x64-1.9.0.tgz
```

## compile and run on IHEP
```bash
# only need cpp, should work well with ROOT
asetup AthAnalysis,21.2.140

# it's not a large project, command line compilation is fine
g++ -O3 -I./dep/onnxruntime/include -I./inc -o run src/main.cpp dep/onnxruntime/lib/libonnxruntime.so

# link to shared library at (pre-)load time
LD_PRELOAD="./dep/onnxruntime/lib/libonnxruntime.so" ./run
```