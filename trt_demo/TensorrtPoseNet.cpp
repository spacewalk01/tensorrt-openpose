#include "TensorrtPoseNet.h"


#define MAX_WORKSPACE (1 << 30) // 1G workspace memory
 
static Logger gLogger;

// Load and deserialize yolo inference engine
TensorrtPoseNet::TensorrtPoseNet(const std::string &engineFilePath, float confThresh, float nmsThresh)
{
	std::cout << "Loading OpenPose Inference Engine ... " << std::endl;


	//cudaSetDevice(1);
	cudaSetDevice(0);
	std::fstream file;

	confThreshold = confThresh;
	nmsThreshold = nmsThresh;

	file.open(engineFilePath, std::ios::binary | std::ios::in);
	if (!file.is_open())
	{
		std::cout << "read engine file: " << engineFilePath << " failed" << std::endl;
		return;
	}
	file.seekg(0, std::ios::end);
	int length = file.tellg();
	file.seekg(0, std::ios::beg);
	std::unique_ptr<char[]> data(new char[length]);
	file.read(data.get(), length);

	file.close();

	auto runtime = UniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
	assert(runtime != nullptr);

	engine = UniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(data.get(), length, nullptr));
	assert(engine != nullptr);

	std::cout << "Done" << std::endl;

	context = UniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
	assert(context);

	const int numBindingPerProfile = engine->getNbBindings() / engine->getNbOptimizationProfiles();
	std::cout << "Number of binding profiles: " << numBindingPerProfile << std::endl;

	initEngine();
}

// Build an yolo inference engine from onnx model
TensorrtPoseNet::TensorrtPoseNet(const std::string &onnxFilePath, int maxBatchSize, float confThresh, float nmsThresh)
{
	std::cout << "Loading OpenPose Onnx Model ... " << std::endl;

	cudaSetDevice(0);

	auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
	assert(builder != nullptr);

	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

	assert(network != nullptr);

	auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
	assert(parser != nullptr);

	if (!parser->parseFromFile(onnxFilePath.c_str(), 2))
	{
		std::string msg("failed to parse onnx file");
		gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
		exit(EXIT_FAILURE);
	}

	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(MAX_WORKSPACE);

	std::cout << "Setting fp16 mode ..." << std::endl;
	if (!builder->platformHasFastFp16())
	{
		std::cout << "Notice: the platform do not have fast fp16" << std::endl;
	}
	else {
		builder->setFp16Mode(true);
		std::cout << "Done." << std::endl;
	}

	std::cout << "Begin building engine..." << std::endl;
	engine = UniquePtr<nvinfer1::ICudaEngine>(builder->buildCudaEngine(*network));

	if (!engine)
	{
		std::string error_message = "Unable to create engine";
		gLogger.log(nvinfer1::ILogger::Severity::kERROR, error_message.c_str());
		exit(-1);
	}
	std::cout << "End building engine..." << std::endl;

	context = UniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
	assert(context);

	initEngine();
}

void TensorrtPoseNet::initEngine()
{
	cudaBuffers.resize(engine->getNbBindings());
	for (size_t i = 0; i < engine->getNbBindings(); ++i)
	{
		auto bindingSize = getSizeByDim(engine->getBindingDimensions(i)) * 1 * sizeof(float);
		cudaMalloc(&cudaBuffers[i], bindingSize);
		if (engine->bindingIsInput(i))
		{
			inputDims.emplace_back(engine->getBindingDimensions(i));
		}
		else
		{
			outputDims.emplace_back(engine->getBindingDimensions(i));
		}
		std::cout << "Binding Name: " << engine->getBindingName(i) << std::endl;
	}
	if (inputDims.empty() || outputDims.empty())
	{
		std::cerr << "Expect at least one input and one output for network";
	}
	
	batchSize = inputDims[0].d[0];
	numChannels = inputDims[0].d[1];
	inputHeightSize = inputDims[0].d[2];
	inputWidthSize = inputDims[0].d[3];

	std::cout << "output[0] -> cmap" << std::endl;
	std::cout << "outputDims[0]: " << std::endl;
	std::size_t size = 1;
	for (std::size_t i = 0; i < outputDims[0].nbDims; ++i)
	{
		std::cout << "out[0]: " << outputDims[0].d[i] << std::endl;
		size *= outputDims[0].d[i];
	}
	std::cout << "out[0].size: " << size << std::endl;

	std::cout << "output[1] -> paf" << std::endl;
	std::cout << "outputDims[1]: " << std::endl;
	size = 1;
	for (std::size_t i = 0; i < outputDims[1].nbDims; ++i)
	{
		std::cout << "out[1]: " << outputDims[1].d[i] << std::endl;
		size *= outputDims[1].d[i];
	}
	std::cout << "out[1].size: " << size << std::endl;

	cpuCmapBuffer.resize(getSizeByDim(outputDims[0]) * batchSize);
	cpuPafBuffer.resize(getSizeByDim(outputDims[1]) * batchSize);

	std::cout << "Model input shape: " <<
		batchSize << "x" << 
		numChannels << "x" << 
		inputWidthSize << "x" << 
		inputHeightSize << std::endl;

	cudaFrame = safeCudaMalloc(4096 * 4096 * 3 * sizeof(uchar)); // max input image shape

	cudaStreamCreate(&cudaStream);
}


void TensorrtPoseNet::infer(cv::Mat &frame)
{
	// Preprocess data and move data to GPU from CPU
	cudaMemcpy(cudaFrame, frame.data, frame.step[0] * frame.rows, cudaMemcpyHostToDevice);
	resizeAndNorm(cudaFrame, (float*)cudaBuffers[0], frame.cols, frame.rows, inputWidthSize, inputHeightSize, cudaStream);
	
	cudaMemcpy(cudaFrame, cudaBuffers[0], getSizeByDim(inputDims[0]) * sizeof(float), cudaMemcpyDeviceToHost);
	
	// Inference
	context->enqueue(batchSize, cudaBuffers.data(), cudaStream, nullptr);

	// Copy results from GPU to CPU	
	cudaMemcpy(cpuCmapBuffer.data(), (float*)cudaBuffers[1], cpuCmapBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuPafBuffer.data(), (float*)cudaBuffers[2], cpuPafBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost);
}

// Serialize the inference engine
bool TensorrtPoseNet::saveEngine(const std::string &engineFilePath)
{
	if (engine)
	{
		std::shared_ptr<nvinfer1::IHostMemory> data(engine->serialize(), InferDeleter());
		std::ofstream file;
		file.open(engineFilePath, std::ios::binary | std::ios::out);
		if (!file.is_open())
		{
			std::cout << "Saving the engine file " << engineFilePath << " failed" << std::endl;
			return 0;
		}
		file.write((const char*)data->data(), data->size());
		file.close();

		std::cout << "Saved engine to " << engineFilePath << std::endl;
	}
	return 1;
}



TensorrtPoseNet::~TensorrtPoseNet() {
	cudaStreamSynchronize(cudaStream);
	cudaStreamDestroy(cudaStream);
	
	for (int i = 0; i < cudaBuffers.size(); ++i) {
		if (cudaBuffers[i])
		{
			cudaFree(cudaBuffers[i]);
		}
	}
}

std::size_t TensorrtPoseNet::getSizeByDim(const nvinfer1::Dims& dims)
{
	std::size_t size = 1;
	for (std::size_t i = 0; i < dims.nbDims; ++i)
	{
		size *= dims.d[i];
	}
	return size;
}