#pragma execution_character_set( "utf-8" )

#include "TensorrtPoseNet.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string> 
#include <map>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Openpose.h"

using namespace cv;
using namespace std::chrono;

static std::vector<std::string> classNames;
static std::vector<Scalar> colors;

int main(int argc, char** argv)
{
	std::string videoPath;
	std::string enginePath = "E:/Programs/TensorRT-7.2.1.6/bin/trt_pose_fp16.engine";

	if (argc > 1)
		videoPath = argv[1];
	else
		videoPath = "E:/Downloads/latin_cardio.mp4";
	
	// Initialize TensorRT
	float confThreshold = 0.3f;
	float nmsThreshold = 0.3f;
	TensorrtPoseNet net(enginePath, confThreshold, nmsThreshold);
	
	// Initialize OpenPose Parser
	Openpose openpose(net.outputDims[0]);
	int* object_counts;
	int* objects;
	float* refined_peaks;

	// Load input video
	VideoCapture cap(videoPath);

	// Write to file
	Size frameSize = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),   
						  (int)cap.get(CAP_PROP_FRAME_HEIGHT));

	std::string outFilename = videoPath.substr(0, videoPath.find("."));

	VideoWriter video;
	video.open(outFilename + "_detected.mp4", -1, 40, frameSize, true);

	if (!cap.isOpened()) {
		std::cout << "Error opening video stream or file" << std::endl;
		return -1;
	}

	float totalRuntime = 0.0, time_span;
	Mat frame, prevFrame;
	high_resolution_clock::time_point t_start, t_end;
	int frameCount = 0;

	while (cap.read(frame))
	{
		t_start = high_resolution_clock::now();

		// Pose Inference
		net.infer(frame);
		openpose.detect(net.cpuCmapBuffer, net.cpuPafBuffer, frame);

		t_end = high_resolution_clock::now();
		time_span = duration_cast<duration<float>>(t_end - t_start).count();
		totalRuntime += time_span;
		frameCount++;

		// Display
		float fps = frameCount / totalRuntime;
		putText(frame, std::to_string((int)fps) + " fps", cv::Point(20, 50), 1.2, 2, cv::Scalar(0, 255, 255), 2);
		putText(frame, "Frame: " + std::to_string(frameCount), cv::Point(20, frame.rows - 35), 1.2, 2, cv::Scalar(0, 255, 255), 2);
		
		video.write(frame);
		imshow("Result", frame);
		waitKey(1);
		if (frameCount > 5000)
			break;
	}

	cap.release();
	video.release();

	return 0;
}
