#include <iostream>
#include <fstream>
#include <iomanip>
#include "movo.h"
#include <DBoW2/DBoW2.h>
#include <DBoW2/ScoringObject.h>
using namespace DBoW2;

double movo::getScale(int frame_id) {
  
  std::string line;
  int i = 0;
  std::ifstream myfile ("../datasets/kitti/poses/00.txt");
  double x =0, y=0, z = 0;
  double x_prev, y_prev, z_prev;
  if (myfile.is_open()) {
    while (( getline (myfile,line) ) && (i<=frame_id)) {
      z_prev = z;
      x_prev = x;
      y_prev = y;
      std::istringstream in(line);
      for (int j=0; j<12; j++)  {
        in >> z ;
        if (j==7) y=z;
        if (j==3)  x=z;
      }
      
      i++;
    }
    myfile.close();
  } else {
    std::cout << "Unable to open file" << std::endl;
    return 0;
  }

  return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;
}

cv::Mat movo::epipolarSearch(std::vector<cv::Point2f> corners1,
					         std::vector<cv::Point2f> corners2,
					         cv::Mat &R, cv::Mat &t) {
	std::vector<cv::Point2f> corners1_ud, corners2_ud;
	undistortPoints(corners1, corners1_ud, K, cv::noArray(), cv::noArray(), cv::noArray());
	undistortPoints(corners2, corners2_ud, K, cv::noArray(), cv::noArray(), cv::noArray());
	cv::Mat mask;
	cv::Mat essMat = findEssentialMat(corners1_ud, corners2_ud, 1.0, cv::Point2d(0.0, 0.0), 
							  cv::RANSAC, 0.99, 
							  5/(K.at<double>(0, 0)+K.at<double>(1, 1)), mask);
	recoverPose(essMat, corners1_ud, corners2_ud, R, t, 1.0, cv::Point2d(0.0, 0.0), mask);
	R = R.inv();
	t = -R*t;
	return mask;
}

void movo::continousOperation() {
	uint database_id = 0;
	uint query_id = database_id + 1; 
	std::vector<cv::Point2f> database_corners;
	cv::Mat database_img, query_img;
	undistort(imread(filenames_left[database_id], CV_8UC1), 
				database_img, K, cv::noArray(), K);


	rows = database_img.rows;
	cols = database_img.cols;

	detectGoodFeatures(database_img, 
					   database_corners,
					   cv::Mat::ones(rows, cols, CV_8UC1));

	cv::Mat rvec, tvec;
	std::vector<cv::Point3f> rvecs, tvecs;
	std::vector<cv::Point2f> query_corners;
	std::vector<cv::Point2f> new_candidate_corners;

	
	cv::Mat traj = cv::Mat::zeros(1000, 1000, CV_8UC3);
	R_global = cv::Mat::eye(3, 3, CV_64FC1);
	t_global = cv::Mat::zeros(3, 1, CV_64FC1);


	// For place recognition
	cv::Ptr<cv::ORB> orb = cv::ORB::create();
    cv::Mat mask_;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<std::vector<cv::Mat > > features;
    orb->detectAndCompute(database_img, mask_, keypoints, descriptors);
    features.push_back(std::vector<cv::Mat >());
    changeStructure(descriptors, features.back());
	while(query_id < filenames_left.size()) {

		undistort(imread(filenames_left[query_id], CV_8UC1), 
					query_img, K, cv::noArray(), K);


	    cv::Mat mask_;
	    std::vector<cv::KeyPoint> keypoints;
	    cv::Mat descriptors;
	    orb->detectAndCompute(query_img, mask_, keypoints, descriptors);
	    features.push_back(std::vector<cv::Mat >());
	    changeStructure(descriptors, features.back());

	    //bool loop_detected = loopDetector(features);

		std::vector<uchar> status1, status2;
		status1 = calculateOpticalFlow(database_img,  query_img,
									  database_corners, query_corners);
		
		filterbyStatus(status1, database_corners, query_corners);
		//std::cout << database_corners.size() << "\t" << query_corners.size() << "\t"<< query_id << std::endl;
		drawmatches(database_img, query_img, database_corners, query_corners);

		cv::Mat mask = epipolarSearch(database_corners, query_corners, R, t);
		double scale = getScale(query_id); 
		//double error = findAvgError(database_corners, query_corners);
		if(scale>0.1 &&
		   fabs(t.at<double>(2)) > fabs(t.at<double>(1)) &&
		   fabs(t.at<double>(2)) > fabs(t.at<double>(0)) &&
		   t.at<double>(2) > 0) {

			t_global = scale*(R_global*t) + t_global;
	    	R_global = R_global*R;
	    	std::cout << t_global.at<double>(0) << std::setw(20) << t_global.at<double>(1) << std::setw(20) 
	    			  << t_global.at<double>(2) << std::endl;
		}
	
		drawTrajectory(t_global, traj);

		cv::Mat mask_mat(query_img.size(), CV_8UC1, cv::Scalar::all(255));
		cv::Mat mask_mat_color;
		cv::cvtColor(mask_mat, mask_mat_color, CV_GRAY2BGR);


		for(int i = 0; i < query_corners.size(); i++) {
			cv::circle(mask_mat_color, query_corners[i], 
				15, CV_RGB(0,0,0), -8, 0);
		}
		cv::cvtColor(mask_mat_color, mask_mat, CV_BGR2GRAY);
		
		detectGoodFeatures(query_img, new_candidate_corners, mask_mat);

		query_corners.insert(query_corners.end(), new_candidate_corners.begin(), 
			new_candidate_corners.end());

		database_corners = query_corners;
		query_img.copyTo(database_img);
		query_id++;

	}
	cv::destroyWindow("img1_l");
	cv::destroyWindow("img2_l");
	cv::destroyWindow("traj");
}

void movo::changeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out) {
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i){
    out[i] = plain.row(i);
  }
}

bool movo::loopDetector(const std::vector<std::vector<cv::Mat > > &features) {
  // branching factor and depth levels 
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  OrbVocabulary voc(k, L, weight, score);
}