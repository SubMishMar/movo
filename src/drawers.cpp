#include <iostream>
#include "movo.h"

void movo::drawmatches(cv::Mat img1, cv::Mat img2, 
				   	   std::vector<cv::Point2f> corners1,
					   std::vector<cv::Point2f> corners2) {
	
	cv::cvtColor(img1, img1_out, CV_GRAY2BGR);
	cv::cvtColor(img2, img2_out, CV_GRAY2BGR);
	for(int l = 0; l < corners1.size(); l++){
		cv::circle(img1_out, corners1[l], 4, CV_RGB(255, 0, 0), -1, 8, 0);
		cv::circle(img2_out, corners2[l], 4, CV_RGB(255, 0, 0), -1, 8, 0);
	}	
	imshow("img0_l", img1_out);
	cv::waitKey(10);
	imshow("img1_l", img2_out);
	cv::waitKey(10);
}

void movo::drawTrajectory(cv::Mat t, cv::Mat &traj) {
    int x = -int(t.at<double>(2)) + 800;
	int y = int(t.at<double>(0)) + 500;
	circle(traj, cv::Point(y, x), 1, CV_RGB(255, 0, 0), 2);
	imshow( "Trajectory", traj);
	cv::waitKey(10);
}

