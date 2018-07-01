#include <iostream>
#include "movo.h"

void movo::detectGoodFeatures(cv::Mat img, 
							  std::vector<cv::Point2f> &corners,
							  cv::Mat mask_mat) {
	cv::TermCriteria termcrit = 
				cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
	goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, mask_mat,
						blockSize, useHarrisDetector, k);

	if(corners.size() > 0)
		cornerSubPix(img, corners, cv::Size(winSizeGFTT/2, winSizeGFTT/2), 
				 	cv::Size(-1, -1), termcrit);
}

std::vector<uchar> movo::calculateOpticalFlow(cv::Mat img1, cv::Mat img2, 
							  				  std::vector<cv::Point2f> &corners1,
							  				  std::vector<cv::Point2f> &corners2) {
	cv::TermCriteria termcrit = cv::TermCriteria(
			cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
	std::vector<uchar> status;
	std::vector<float> err;
	calcOpticalFlowPyrLK(img1, img2, corners1, corners2, status, err, 
						 cv::Size(winSizeGFTT + 1, winSizeGFTT + 1), 
						 3, termcrit, 0, 0.001); 	
	return status;
}