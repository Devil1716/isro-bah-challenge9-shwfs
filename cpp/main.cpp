// Real-time TensorRT inference core for Shack-Hartmann wavefront reconstruction.
//
// Build on Ubuntu with CUDA + TensorRT installed:
//   mkdir -p build && cd build
//   cmake .. && cmake --build . -j
//
// Run:
//   ./shwfs_trt wavefront_net_fp16.engine 128 128 16384
//
// Arguments:
//   1: TensorRT serialized engine path
//   2: input height
//   3: input width
//   4: output phase element count, e.g. 128*128 = 16384
//
// This expects the ONNX exported by shwfs_pipeline.py to be converted with:
//   trtexec --onnx=wavefront_net.onnx --saveEngine=wavefront_net_fp16.engine --fp16

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

#define CHECK_CUDA(expr)                                                                            \
    do {                                                                                            \
        cudaError_t status = (expr);                                                                \
        if (status != cudaSuccess) {                                                                \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status));    \
        }                                                                                           \
    } while (0)

class TrtLogger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << '\n';
        }
    }
};

template <typename T>
struct TrtDestroy {
    void operator()(T* ptr) const {
        if (ptr) {
            delete ptr;
        }
    }
};

template <typename T>
using TrtUnique = std::unique_ptr<T, TrtDestroy<T>>;

std::vector<char> readBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + path);
    }
    const auto size = file.tellg();
    std::vector<char> buffer(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), size);
    return buffer;
}

size_t volume(const nvinfer1::Dims& dims) {
    size_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            throw std::runtime_error("Dynamic dimension encountered. Build a fixed or profiled engine.");
        }
        v *= static_cast<size_t>(dims.d[i]);
    }
    return v;
}

size_t elementSize(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            throw std::runtime_error("Unsupported TensorRT data type.");
    }
}

}  // namespace

class WavefrontTrtEngine {
  public:
    WavefrontTrtEngine(const std::string& enginePath, int inputH, int inputW, int phaseElements)
        : inputH_(inputH), inputW_(inputW), phaseElements_(phaseElements) {
        auto engineData = readBinaryFile(enginePath);

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }

        engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize TensorRT engine.");
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Failed to create TensorRT execution context.");
        }

        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        allocateBindings();
    }

    ~WavefrontTrtEngine() {
        for (void* ptr : deviceBindings_) {
            cudaFree(ptr);
        }
        if (hostInput_) {
            cudaFreeHost(hostInput_);
        }
        if (hostPhaseOutput_) {
            cudaFreeHost(hostPhaseOutput_);
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    WavefrontTrtEngine(const WavefrontTrtEngine&) = delete;
    WavefrontTrtEngine& operator=(const WavefrontTrtEngine&) = delete;

    float* hostInput() {
        return hostInput_;
    }

    const float* infer(const float* inputFrame) {
        const size_t inputBytes = inputElements_ * sizeof(float);
        CHECK_CUDA(cudaMemcpyAsync(
            deviceBindings_[inputBinding_], inputFrame, inputBytes, cudaMemcpyHostToDevice, stream_));

        bool ok = enqueue();
        if (!ok) {
            throw std::runtime_error("TensorRT enqueue failed.");
        }

        CHECK_CUDA(cudaMemcpyAsync(
            hostPhaseOutput_,
            deviceBindings_[phaseOutputBinding_],
            phaseElements_ * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_));
        CHECK_CUDA(cudaStreamSynchronize(stream_));
        return hostPhaseOutput_;
    }

    double benchmark(int warmupIters = 100, int timedIters = 1000) {
        std::fill(hostInput_, hostInput_ + inputElements_, 0.0f);
        // Synthetic normalized spot frame. In deployment, write camera DMA or
        // preprocessing output into hostInput_ or directly into device input.
        for (size_t i = 0; i < inputElements_; i += 17) {
            hostInput_[i] = 1.0f;
        }

        for (int i = 0; i < warmupIters; ++i) {
            infer(hostInput_);
        }

        cudaEvent_t start{}, stop{};
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, stream_));
        for (int i = 0; i < timedIters; ++i) {
            const size_t inputBytes = inputElements_ * sizeof(float);
            CHECK_CUDA(cudaMemcpyAsync(
                deviceBindings_[inputBinding_], hostInput_, inputBytes, cudaMemcpyHostToDevice, stream_));
            if (!enqueue()) {
                throw std::runtime_error("TensorRT enqueue failed during benchmark.");
            }
        }
        CHECK_CUDA(cudaEventRecord(stop, stream_));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        return static_cast<double>(elapsedMs) / static_cast<double>(timedIters);
    }

  private:
    bool enqueue() {
#if NV_TENSORRT_MAJOR >= 10
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (!context_->setTensorAddress(name, deviceBindings_[i])) {
                return false;
            }
        }
        return context_->enqueueV3(stream_);
#else
        return context_->enqueueV2(deviceBindings_.data(), stream_, nullptr);
#endif
    }

    void allocateBindings() {
#if NV_TENSORRT_MAJOR >= 10
        const int nb = engine_->getNbIOTensors();
        deviceBindings_.resize(nb, nullptr);
        for (int i = 0; i < nb; ++i) {
            const char* name = engine_->getIOTensorName(i);
            const auto mode = engine_->getTensorIOMode(name);
            auto dtype = engine_->getTensorDataType(name);

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                inputBinding_ = i;
                nvinfer1::Dims dims = nvinfer1::Dims4{1, 1, inputH_, inputW_};
                if (!context_->setInputShape(name, dims)) {
                    throw std::runtime_error("Failed to set TensorRT input shape.");
                }
            }
        }

        for (int i = 0; i < nb; ++i) {
            const char* name = engine_->getIOTensorName(i);
            const auto mode = engine_->getTensorIOMode(name);
            nvinfer1::Dims dims = context_->getTensorShape(name);
            auto dtype = engine_->getTensorDataType(name);
            if (dtype != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error("This sample expects FP32 I/O bindings. Keep TensorRT outputs FP32.");
            }

            const size_t bytes = volume(dims) * elementSize(dtype);
            CHECK_CUDA(cudaMalloc(&deviceBindings_[i], bytes));

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                inputElements_ = volume(dims);
            } else if (std::string(name).find("phase") != std::string::npos || phaseOutputBinding_ < 0) {
                phaseOutputBinding_ = i;
            }
        }
#else
        const int nb = engine_->getNbBindings();
        deviceBindings_.resize(nb, nullptr);
        for (int i = 0; i < nb; ++i) {
            nvinfer1::Dims dims = engine_->getBindingDimensions(i);
            auto dtype = engine_->getBindingDataType(i);

            if (engine_->bindingIsInput(i)) {
                inputBinding_ = i;
                dims = nvinfer1::Dims4{1, 1, inputH_, inputW_};
                if (!context_->setBindingDimensions(i, dims)) {
                    throw std::runtime_error("Failed to set TensorRT input binding dimensions.");
                }
            }
        }

        for (int i = 0; i < nb; ++i) {
            nvinfer1::Dims dims = context_->getBindingDimensions(i);
            auto dtype = engine_->getBindingDataType(i);
            if (dtype != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error("This sample expects FP32 I/O bindings. Keep TensorRT outputs FP32.");
            }

            const size_t bytes = volume(dims) * elementSize(dtype);
            CHECK_CUDA(cudaMalloc(&deviceBindings_[i], bytes));

            if (engine_->bindingIsInput(i)) {
                inputElements_ = volume(dims);
            } else if (phaseOutputBinding_ < 0) {
                phaseOutputBinding_ = i;
            }
        }
#endif

        if (inputBinding_ < 0 || phaseOutputBinding_ < 0) {
            throw std::runtime_error("Could not identify input and phase output bindings.");
        }

        CHECK_CUDA(cudaHostAlloc(
            reinterpret_cast<void**>(&hostInput_), inputElements_ * sizeof(float), cudaHostAllocPortable));
        CHECK_CUDA(cudaHostAlloc(
            reinterpret_cast<void**>(&hostPhaseOutput_), phaseElements_ * sizeof(float), cudaHostAllocPortable));
    }

    TrtLogger logger_{};
    TrtUnique<nvinfer1::IRuntime> runtime_{nullptr};
    TrtUnique<nvinfer1::ICudaEngine> engine_{nullptr};
    TrtUnique<nvinfer1::IExecutionContext> context_{nullptr};
    cudaStream_t stream_{};
    std::vector<void*> deviceBindings_{};
    float* hostInput_{nullptr};
    float* hostPhaseOutput_{nullptr};
    int inputBinding_{-1};
    int phaseOutputBinding_{-1};
    int inputH_{0};
    int inputW_{0};
    int phaseElements_{0};
    size_t inputElements_{0};
};

int main(int argc, char** argv) {
    try {
        if (argc < 5) {
            std::cerr << "Usage: " << argv[0] << " <engine.trt> <input_h> <input_w> <phase_elements>\n";
            return 2;
        }

        const std::string enginePath = argv[1];
        const int inputH = std::stoi(argv[2]);
        const int inputW = std::stoi(argv[3]);
        const int phaseElements = std::stoi(argv[4]);

        WavefrontTrtEngine engine(enginePath, inputH, inputW, phaseElements);

        // Replace this synthetic frame with the latest normalized SHWFS detector
        // frame. For lowest latency, make camera/preprocessing write directly to
        // the device input buffer or use CUDA pinned host memory.
        float* frame = engine.hostInput();
        std::fill(frame, frame + static_cast<size_t>(inputH * inputW), 0.0f);
        for (int y = 0; y < inputH; y += 8) {
            for (int x = 0; x < inputW; x += 8) {
                frame[y * inputW + x] = 1.0f;
            }
        }

        const float* phase = engine.infer(frame);
        std::cout << "phase[0:8] = ";
        for (int i = 0; i < std::min(8, phaseElements); ++i) {
            std::cout << phase[i] << ' ';
        }
        std::cout << '\n';

        const double avgMs = engine.benchmark();
        std::cout << "Average GPU enqueue time: " << avgMs << " ms/frame\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal: " << ex.what() << '\n';
        return 1;
    }
}
