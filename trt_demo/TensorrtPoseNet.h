#pragma once
#include <string>
#include <memory>
#include <vector>
#include <fstream>
#include "NvInfer.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include "resize.h"

enum class RUN_MODE
{
	FLOAT32 = 0,
	FLOAT16 = 1,
	INT8 = 2
};

struct InferDeleter
{
	template <typename T>
	void operator()(T* obj) const
	{
		if (obj)
		{
			obj->destroy();
		}
	}
};


inline void* safeCudaMalloc(size_t memSize)
{
	void* deviceMem;
	cudaMalloc(&deviceMem, memSize);
	if (deviceMem == nullptr)
	{
		std::cerr << "Out of memory" << std::endl;
		exit(1);
	}
	return deviceMem;
}



class Logger : public nvinfer1::ILogger
{
public:

	void log(Severity severity, const char* msg) override 
	{
		if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) 
		{
			std::cerr << msg << std::endl;
		}
	}
};


class TensorrtPoseNet
{
	template <typename T>
	using UniquePtr = std::unique_ptr<T, InferDeleter>;

public: 

	TensorrtPoseNet(const std::string &onnxFilePath, int maxBatchSize, float confThresh, float nmsThresh);
	TensorrtPoseNet(const std::string &engineFilePath, float confThresh, float nmsThresh);
	
	void infer(cv::Mat &img);	

	bool saveEngine(const std::string &engineFilePath);
	~TensorrtPoseNet();

	// The dimensions of the input and output to the network

	int batchSize;
	int numClasses;
	int numChannels;
	int inputHeightSize;
	int inputWidthSize;
	
	std::vector<float> cpuCmapBuffer;
	std::vector<float> cpuPafBuffer;

	std::vector<nvinfer1::Dims> inputDims;
	std::vector<nvinfer1::Dims> outputDims;


private:
	std::size_t getSizeByDim(const nvinfer1::Dims& dims);
	void preprocessImage(cv::Mat &frame, float* gpu_input);

	void initEngine();

	UniquePtr<nvinfer1::ICudaEngine> engine;
	UniquePtr<nvinfer1::IExecutionContext> context;
	cudaStream_t cudaStream;
	
	std::vector<void*> cudaBuffers;
	void *cudaFrame;

	float confThreshold = 0.4f;
	float nmsThreshold = 0.4f;


};