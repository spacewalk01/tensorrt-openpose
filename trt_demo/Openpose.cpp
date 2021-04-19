#include "Openpose.h"
  
void Openpose::detect(std::vector<float> cmap_vector, std::vector<float> paf_vector, cv::Mat &frame) 
{
	/*
	 Input arguments:
		cmap: feature maps of joints
		paf: connections between joints
		frame: image data

	 output arguments:
		object_counts_ptr[0];			// N
		objects_ptr[0];					// NxMxC
		refined_peaks_ptr[0];			// NxCxMx2
	*/

	// ****** DETECT SKELETON ***** //

	// 1. Find peaks (NMS)
	
	const float *input_ptr = &cmap_vector[0];

	std::vector<int> peaks, peak_counts;
	size_t peak_size = N * C * M * 2;	// NxCxMx2
	peaks.resize(peak_size);
	size_t peak_count_size = N * C; // NxC
	peak_counts.resize(peak_count_size);
	
	int* peaks_ptr = &peaks[0];			// return value
	int* peak_counts_ptr = &peak_counts[0];	// return value 
	 
	trt_pose::parse::find_peaks_out_nchw(peak_counts_ptr, peaks_ptr, input_ptr, N, C, H, W, M, cmap_threshold, cmap_window);
	  
	// 2. Refine peaks

	std::vector<float> refined_peaks;		// NxCxMx2
	refined_peaks.resize(peak_size); 
	for (int i = 0; i < refined_peaks.size(); i++) {
		refined_peaks[0] = 0;
	}
	float *refined_peaks_ptr = &refined_peaks[0]; // return value
	 
	trt_pose::parse::refine_peaks_out_nchw(refined_peaks_ptr, peak_counts_ptr, peaks_ptr, input_ptr, N, C, H, W, M, cmap_window);
 
	// 3. Score paf 
	const float *paf_ptr = &paf_vector[0];

	int K = 21;

	size_t score_graph_size = N * K * M * M;	// NxKxMxM
	std::vector<float> score_graph;
	score_graph.resize(score_graph_size);
	float* score_graph_ptr = &score_graph[0];
	 
	trt_pose::parse::paf_score_graph_out_nkhw(score_graph_ptr, topology, paf_ptr, peak_counts_ptr, refined_peaks_ptr,
		N, K, C, H, W, M, line_integral_samples);

	// 4. Assignment algorithm

	std::vector<int> connections;
	connections.resize(N * K * 2 * M);
	for (int i = 0; i < connections.size(); i++)
	{
		connections[i] = -1;
	}

	int* connections_ptr = &connections[0];

	void *workspace = (void *)malloc(trt_pose::parse::assignment_out_workspace(M));
	 
	trt_pose::parse::assignment_out_nk(connections_ptr, score_graph_ptr, topology, peak_counts_ptr, N, C, K, M, link_threshold, workspace);
	
	free(workspace);

	// 5. Merging
	
	std::vector<int> objects;
	objects.resize(N * max_num_objects * C);
	for (int i = 0; i < objects.size(); i++) {
		objects[i] = -1;
	}
	int *objects_ptr = &objects[0];

	std::vector<int> object_counts;
	object_counts.resize(N);
	object_counts[0] = 0;	// batchSize=1		
	int *object_counts_ptr = &object_counts[0];

	void *merge_workspace = malloc(trt_pose::parse::connect_parts_out_workspace(C, M));

	trt_pose::parse::connect_parts_out_batch(object_counts_ptr, objects_ptr, connections_ptr, topology, peak_counts_ptr, N, K, C, M, max_num_objects, merge_workspace);
	
	free(merge_workspace);

	// ****** DRAWING SKELETON ***** //
	
	for (int i = 0; i < object_counts[0]; i++) {
		
		int *obj = &objects_ptr[C * i];

		for (int j = 0; j < C; j++) {

			int k = (int)obj[j];
			if (k >= 0) {
				float *peak = &refined_peaks_ptr[j * M * 2];
				int x = (int)(peak[k * 2 + 1] * frame.cols);
				int y = (int)(peak[k * 2] * frame.rows);
				circle(frame, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), cv::FILLED, 4, 0);
			}
		} 
		
		for (int k = 0; k < K; k++) {
			int c_a = topology[k * 4 + 2];
			int c_b = topology[k * 4 + 3];

			if (obj[c_a] >= 0 && obj[c_b] >= 0) {
				float *peak0 = &refined_peaks_ptr[c_a * M * 2];
				float *peak1 = &refined_peaks_ptr[c_b * M * 2];

				int x0 = (int)(peak0[(int)obj[c_a] * 2 + 1] * frame.cols);
				int y0 = (int)(peak0[(int)obj[c_a] * 2] * frame.rows);
				int x1 = (int)(peak1[(int)obj[c_b] * 2 + 1] * frame.cols);
				int y1 = (int)(peak1[(int)obj[c_b] * 2] * frame.rows);
				line(frame, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 2, 1);
			}
		}
	}
}
		
