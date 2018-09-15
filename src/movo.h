#include <eigen3/Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

class movo{
private:
	// for GFTT 
	int maxCorners;
  	double qualityLevel;
  	double minDistance;
  	int blockSize;
  	bool useHarrisDetector;
	double k ;
	int winSizeGFTT;

	// Calibration Matrix
	cv::Mat P_L, K;

	// for FAST
    int fast_threshold;
    bool nonmaxSuppression;
    int winSizeFAST;


    // folder and file names
    std::vector<cv::String> filenames_left;
    cv::String folder_left;
    std::string config_file; 

    // Image containers and related Mats and Vectors
    cv::Mat img1_out, img2_out;

    // pose
    cv::Mat R, t, R_global, t_global;

    //
    double rows, cols;

public:
	//construtor
	movo(int argc, char **argv) {
		readParams(argc, argv);
		K = P_L(cv::Range(0,3), cv::Range(0, 3));
	}

	//reads Params related to all functionalities
	void readParams(int, char**);

	// detects gftt
	void detectGoodFeatures(cv::Mat img, 
							std::vector<cv::Point2f> &corners,
							cv::Mat mask_mat);

	//Caluclates optical flow and returns a status to represent points
	//for which a valied tracked point has been found
	std::vector<uchar> calculateOpticalFlow(cv::Mat img1, cv::Mat img2, 
							  				std::vector<cv::Point2f> &corners1,
							  				std::vector<cv::Point2f> &corners2);

	//Estimates R, t from tracked feature points and returns an inlier mask
	cv::Mat epipolarSearch(std::vector<cv::Point2f> corners1,
					       std::vector<cv::Point2f> corners2,
					       cv::Mat &R, cv::Mat &t);

	void filterbyStatus(std::vector<uchar> status,
					    std::vector<cv::Point2f> &corners1,
					    std::vector<cv::Point2f> &corners2);

	//Drawmatches
	void drawmatches(cv::Mat img1, cv::Mat img2, 
					 std::vector<cv::Point2f> corners1,
					 std::vector<cv::Point2f> corners2);



	//Continous VO operation
	void continousOperation();



	//Draw trajectory;
	void drawTrajectory(cv::Mat t, cv::Mat &traj);


	double getScale(int frame_id);
};
