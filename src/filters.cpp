#include <iostream>
#include "movo.h"

void movo::filterbyMask(cv::Mat mask,
					    std::vector<cv::Point2f> &corners1,
					    std::vector<cv::Point2f> &corners2) {
	int j = 0;
	for(int i = 0; i < corners1.size(); i++) {
		if(!mask.at<unsigned char>(i)) {
			continue;
		}
		corners1[j] = corners1[i];
		corners2[j] = corners2[i];
		j++;
	}
	corners1.resize(j);
	corners2.resize(j);	
}

void movo::filterbyStatus(std::vector<uchar> status,
					      std::vector<cv::Point2f> &corners1,
					      std::vector<cv::Point2f> &corners2) {
	size_t j = 0;
	for(size_t i = 0; i < status.size(); i++) {
		if((int)status[i] == 0 ||
		   corners2[i].x < 0 || corners2[i].y < 0 ||
		   corners2[i].x > cols || corners2[i].y > rows) {
		   	continue;
		} 
		corners1[j] = corners1[i];
		corners2[j] = corners2[i];
		j++;
	}
	corners1.resize(j);
	corners2.resize(j);	
}
