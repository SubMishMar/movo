#include <iostream>
#include "movo.h"

cv::Mat movo::vector2mat(cv::Point2f pt2d) {
	cv::Mat pt2d_mat=cv::Mat::zeros(2, 1, CV_64FC1);
	pt2d_mat.at<double>(0) = pt2d.x;
	pt2d_mat.at<double>(1) = pt2d.y;
	return pt2d_mat;
}

cv::Mat movo::vector2mat(cv::Point3f pt3d) {
	cv::Mat pt3d_mat=cv::Mat::zeros(3, 1, CV_64FC1);
	pt3d_mat.at<double>(0) = pt3d.x;
	pt3d_mat.at<double>(1) = pt3d.y;
	pt3d_mat.at<double>(2) = pt3d.z;
	return pt3d_mat;
}

void movo::convertFromHomogeneous(cv::Mat p3h, std::vector<cv::Point3f> &p3uh) {
	p3uh.clear();
	for(int i = 0; i < p3h.cols; i++) {
	  cv::Mat p3d_col_i;
	  cv::Mat p3h_col_i = p3h.col(i);
	  convertPointsFromHomogeneous(p3h_col_i.t(), p3d_col_i);
	  float x = (float)p3d_col_i.at<double>(0);
	  float y = (float)p3d_col_i.at<double>(1);
	  float z = (float)p3d_col_i.at<double>(2);
	  p3uh.push_back(cv::Point3f(x, y, z));
	}
}

void movo::corners2keypoint(std::vector<cv::Point2f> corners,
					  std::vector<keypoint> &keypoints,
					  int frame,
					  cv::Mat M) {
	for(int i = 0; i < corners.size(); i++) {
		keypoint kpt;
		kpt.pt = corners[i];
		kpt.id0 = frame;
		kpt.M0 = M;
		keypoints.push_back(kpt); 
	}
}