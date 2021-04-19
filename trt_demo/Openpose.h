#pragma once

#include "parse/refine_peaks.hpp"
#include "parse/find_peaks.hpp"
#include "parse/paf_score_graph.hpp"
#include "parse/munkres.hpp" 
#include "parse/connect_parts.hpp" 
#include <vector>
#include <iostream>
#include "NvInfer.h"
#include <stdlib.h>
#include <opencv2/opencv.hpp>

class Openpose
{

public:
	Openpose(nvinfer1::Dims inputDims) {

		cmap_threshold = 0.1;
		link_threshold = 0.1;
		cmap_window = 5;
		line_integral_samples = 7;
		max_num_objects = 100;

		N = inputDims.d[0];
		C = inputDims.d[1];
		H = inputDims.d[2];
		W = inputDims.d[3];
		M = 100;

		//int N = 1;
		//int C = 18;
		//int H = 56;
		//int W = 56;
	};
	void detect(std::vector<float> cmap_vector, std::vector<float> paf_vector, cv::Mat &frame);


private:
	int topology[84] = { 0, 1, 15, 13, 2, 3, 13, 11, 4, 5, 16, 14, 6, 7, 14, 12, 8, 9, 11, 12, 10, 11, 5, 7, 12, 13, 6, 8, 14, 15, 7, 9, 16, 17, 8, 10, 18, 19, 1, 2, 20, 21, 0, 1, 22, 23, 0, 2, 24, 25, 1, 3, 26, 27, 2, 4, 28, 29, 3, 5, 30, 31, 4, 6, 32, 33, 17, 0, 34, 35, 17, 5, 36, 37, 17, 6, 38, 39, 17, 11, 40, 41, 17, 12};

	float cmap_threshold;
	float link_threshold;
	int cmap_window;
	int line_integral_samples;
	int max_num_objects;

	int N;
	int C;
	int H;
	int W;
	int M;
};

