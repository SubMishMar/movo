#include <iostream>
#include <ros/ros.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>


#include <sensor_msgs/ChannelFloat32.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>

typedef message_filters::sync_policies::ApproximateTime
        <sensor_msgs::Image,
         sensor_msgs::PointCloud> SyncPolicy;

struct pointWiD {
    cv::Point2d point_xy;
    cv::Point2d point_uv;
    int id;
    int ref_frame;
};

struct mapPoint {
    cv::Point3d pointXYZ; // X_i : 3D pt in observation frame
    cv::Point2d point_uv; // pixel value in captured frame
    cv::Point2d point_xy; // normalized coordinates in the captured frame
    int id;
    int ref_frame;
    Eigen::Matrix4d T_obspose;
};

struct feature_id_pose {
    cv::Point2d pt_xy; // normalized feaure loc in image plane
    cv::Point2d pt_uv;
    int id;
    Eigen::Matrix4d obs_pose; // pose at which pt_xy was observed
    int image_frame_no;
};

struct point3DWiD {
    cv::Point3d point_3d;
    int id;
};

class Odometry {
private:
    ros::NodeHandle n;

    message_filters::Subscriber<sensor_msgs::Image> *image_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud> *cloud_sub;
    message_filters::Synchronizer<SyncPolicy> *sync;

    std::vector<pointWiD> curr_features;
    std::vector<int> curr_idx;
    std::vector<pointWiD> prev_features;
    std::vector<int> prev_idx;
    std::vector<pointWiD> init_features;
    std::vector<int> init_idx;

    std::vector<int> common_ids;

    std::string config_file;
    std::string image_topic;

    bool first_frame;
    bool initialized;
    bool view_initialization;

    double fx, fy, cx, cy;
    double k1, k2, p1, p2, k3;

    cv::Mat K_, D_;
    cv::Mat init_image;
    cv::Mat curr_image;

    std::vector<mapPoint> map_points_id;

    std::vector<point3DWiD> _3dPoints_k_1;
    std::vector<point3DWiD> _3dPoints_k;
    int frame_no;

    Eigen::Matrix4d w_T_i;

    std::vector<feature_id_pose>  triangulation_candidates;

    cv::Mat traj;
    double relative_scale;
    std::string sensor_name;

    double max_dist;
    double rep_err;

public:
    Odometry(ros::NodeHandle nh) {
        n = nh;

        config_file = readParam<std::string>(n, "config_file");
        cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
        if(!fsSettings.isOpened()) {
            ROS_ERROR_STREAM("Wrong path to settings");
        }
        fsSettings["image_topic"] >> image_topic;
        fsSettings["sensor_name"] >> sensor_name;
        fsSettings["max_dist"] >> max_dist;
        fsSettings["rep_err"] >> rep_err;

        ROS_INFO_STREAM("Image Topic: " << image_topic);
        ROS_INFO_STREAM("Sensor Name: " << sensor_name);
        ROS_INFO_STREAM("Max Triangulation Dist: " << max_dist);
        ROS_INFO_STREAM("Rep Err for PnP: " << rep_err);
        image_sub = new
                message_filters::Subscriber
                        <sensor_msgs::Image>(n, image_topic, 1);
        cloud_sub = new
                message_filters::Subscriber
                        <sensor_msgs::PointCloud>(n, "/feature_tracker/feature", 1);
        sync = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10),
                                                             *image_sub,
                                                             *cloud_sub);
        sync->registerCallback(boost::bind(&Odometry::callback, this, _1, _2));


        K_ = cv::Mat::zeros(3, 3, CV_64F);
        D_ = cv::Mat::zeros(1, 5, CV_64F);

        cv::FileNode n1 = fsSettings["projection_parameters"];
        K_.at<double>(0, 0) = fx = static_cast<double>(n1["fx"]);
        K_.at<double>(0, 2) = cx = static_cast<double>(n1["cx"]);
        K_.at<double>(1, 1) = fy = static_cast<double>(n1["fy"]);
        K_.at<double>(1, 2) = cy = static_cast<double>(n1["cy"]);
        K_.at<double>(2, 2) = 1;

        cv::FileNode n2 = fsSettings["distortion_parameters"];
        D_.at<double>(0) = k1 = static_cast<double>(n2["k1"]);
        D_.at<double>(1) = k2 = static_cast<double>(n2["k2"]);
        D_.at<double>(2) = p1 = static_cast<double>(n2["p1"]);
        D_.at<double>(3) = p2 = static_cast<double>(n2["p2"]);
        D_.at<double>(3) = k3 = static_cast<double>(n2["k3"]);

        std::cout << K_ << std::endl;
        std::cout << D_ << std::endl;
        first_frame = true;
        initialized = false;
        view_initialization = false; // this is for debugging
        frame_no = 0;

        w_T_i = Eigen::Matrix4d::Identity();

        traj = cv::Mat::zeros(1000, 1000, CV_8UC3);
        cv::namedWindow("Visual Odometry Tracking");
        relative_scale = 1;
    }

    template <typename T>
    T readParam(ros::NodeHandle &n, std::string name) {
        T ans;
        if (n.getParam(name, ans)) {
            ROS_INFO_STREAM("Loaded " << name << ": " << ans);
        }
        else {
            ROS_ERROR_STREAM("Failed to load " << name);
            n.shutdown();
        }
        return ans;
    }

    void viewTrajectory(Eigen::Vector3d XYZ) {
        int x = -int(XYZ.z()) + 800;
        int y = int(XYZ.x()) + 500;
        circle(traj, cv::Point(y, x), 1, CV_RGB(255, 0, 0), 2);
        imshow( "Visual Odometry Trajectory", traj);
        cv::waitKey(1);
    }

    void viewInitialization(std::vector<cv::Point3d> point3d_homo,
            cv::Mat rot, cv::Mat trans,
            std::vector<cv::Point2d> good_features1,
            std::vector<cv::Point2d> good_features2) {

        assert(good_features1.size() ==  good_features2.size());
        for (int i = 0; i < good_features1.size(); i++) {
            cv::circle(init_image, good_features1[i], 2,
                    cv::Scalar(255, 255,  0), 10);
            cv::circle(curr_image, good_features2[i], 2,
                       cv::Scalar(255, 255,  0), 10);
        }

        cv::resize(init_image, init_image, cv::Size(), 0.35, 0.35);
        cv::resize(curr_image, curr_image, cv::Size(), 0.35, 0.35);

        cv::imshow("init image", init_image);
        cv::imshow("curr image", curr_image);
        cv::waitKey(-1);
        pcl::visualization::PCLVisualizer viewer("Viewer");
        viewer.setBackgroundColor (255, 255, 255);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud->points.resize (point3d_homo.size());
        for(int i = 0; i < point3d_homo.size(); i++) {
            pcl::PointXYZRGB &point = cloud->points[i];
            point.x = point3d_homo[i].x;
            point.y = point3d_homo[i].y;
            point.z = point3d_homo[i].z;
            point.r = 0;
            point.g = 0;
            point.b = 255;
        }
        viewer.addPointCloud(cloud, "Triangulated Point Cloud");
        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 3,
                                                 "Triangulated Point Cloud");
        viewer.addCoordinateSystem (1.0);
        // add the second camera pose
        Eigen::Matrix4f eig_mat;
        Eigen::Affine3f cam_pose;

        rot.convertTo(rot, CV_32F);
        trans.convertTo(trans, CV_32F);

        //this shows how a camera moves
        cv::Mat Rinv = rot.t();
        cv::Mat T = -Rinv * trans;

        eig_mat(0,0) =
                Rinv.at<float>(0,0);eig_mat(0,1) = Rinv.at<float>(0,1);eig_mat(0,2) = Rinv.at<float>(0,2);
        eig_mat(1,0) =
                Rinv.at<float>(1,0);eig_mat(1,1) = Rinv.at<float>(1,1);eig_mat(1,2) = Rinv.at<float>(1,2);
        eig_mat(2,0) =
                Rinv.at<float>(2,0);eig_mat(2,1) = Rinv.at<float>(2,1);eig_mat(2,2) = Rinv.at<float>(2,2);
        eig_mat(3,0) =
                0.f; eig_mat(3,1) = 0.f; eig_mat(3,2) = 0.f;
        eig_mat(0, 3) =
                T.at<float>(0);
        eig_mat(1, 3) =
                T.at<float>(1);
        eig_mat(2, 3) =
                T.at<float>(2);
        eig_mat(3, 3) = 1.f;

        cam_pose = eig_mat;

        //cam_pose should be Affine3f, Affine3d cannot be used
        viewer.addCoordinateSystem(1.0, cam_pose, "2nd cam");

        viewer.initCameraParameters ();
        while (!viewer.wasStopped ()) {
            viewer.spin();
        }
        ros::shutdown();
    }

    std::vector<int> getMatchIds(std::vector<int> id_list1,
                                 std::vector<int> id_list2) {
        std::sort(id_list1.begin(), id_list1.end());
        std::sort(id_list2.begin(), id_list2.end());
        // Find common ids
        std::vector<int> common_ids(std::min(id_list1.size(), id_list2.size()));
        std::vector<int>::iterator it, st;
        it = set_intersection(id_list1.begin(),
                              id_list1.end(),
                              id_list2.begin(),
                              id_list2.end(),
                              common_ids.begin());
        std::vector<int> common_indices_1;
        for (st = common_ids.begin(); st != it; ++st) {
            common_indices_1.push_back(*st);
        }
        return common_indices_1;
    }

    void initialize() {
        ROS_WARN_STREAM("Initializing");
        std::vector<int> common_indices =
                getMatchIds(init_idx, curr_idx);
        std::vector<cv::Point2d> init_pts;
        std::vector<cv::Point2d> curr_pts;
        std::vector<cv::Point2d> init_pts_uv;
        std::vector<cv::Point2d> curr_pts_uv;
        // Find common ids
        for(int count = 0; count < common_indices.size(); count++) {
            int query_id = common_indices[count];
            for(int i = 0; i < init_features.size(); i++) {
                if(query_id == init_features[i].id){
                    init_pts.push_back(init_features[i].point_xy);
                    init_pts_uv.push_back(init_features[i].point_uv);
                    break;
                }
            }
            for(int i = 0; i < curr_features.size(); i++) {
                if(query_id == curr_features[i].id){
                    curr_pts.push_back(curr_features[i].point_xy);
                    curr_pts_uv.push_back(curr_features[i].point_uv);
                    break;
                }
            }
        }

        int no_of_matches = init_pts.size();
//        ROS_INFO_STREAM("No of matches: " << no_of_matches);
        double sum = 0;
        for(int i = 0; i < no_of_matches; i++) {
            cv::Point2d difference = init_pts[i] - curr_pts[i];
            sum += sqrt(difference.dot(difference));
        }
        sum /=  no_of_matches;

        cv::Mat mask;
        cv::Mat E = cv::findFundamentalMat(init_pts, curr_pts,
                                           cv::FM_RANSAC, 0.3/460, 0.99, mask);
        std::vector<cv::Point2d> inlier_match_points1, inlier_match_points2;
        std::vector<cv::Point2d> inlier_match_points1_uv, inlier_match_points2_uv;
        std::vector<int> inlier_ids;
        for(int i = 0; i < mask.rows; i++) {
            if(mask.at<unsigned char>(i)){
                inlier_match_points1.push_back(init_pts[i]);
                inlier_match_points2.push_back(curr_pts[i]);
                inlier_match_points1_uv.push_back(init_pts_uv[i]);
                inlier_match_points2_uv.push_back(curr_pts_uv[i]);
                inlier_ids.push_back(common_indices[i]);
            }
        }

        cv::Mat cameraMatrix =
                (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot, trans;
        mask.release();
        int inlier_cnt = cv::recoverPose(E,
                                         inlier_match_points1,
                                         inlier_match_points2,
                                         cameraMatrix,
                                         rot, trans, mask);
        if(sum*460 > 30 || inlier_cnt > 50) {
            std::vector<cv::Point2d> triangulation_points1, triangulation_points2;
            std::vector<cv::Point2d> triangulation_points1_uv, triangulation_points2_uv;
            std::vector<int> triangulation_ids;
            for(int i = 0; i < mask.rows; i++) {
                if(mask.at<unsigned char>(i)){
                    triangulation_points1.push_back(inlier_match_points1[i]);
                    triangulation_points2.push_back(inlier_match_points2[i]);
                    triangulation_points1_uv.push_back(inlier_match_points1_uv[i]);
                    triangulation_points2_uv.push_back(inlier_match_points2_uv[i]);
                    triangulation_ids.push_back(inlier_ids[i]);
                }
            }

            cv::Mat Rt0 = cv::Mat::eye(3, 4, CV_64FC1);
            cv::Mat Rt1 = cv::Mat::eye(3, 4, CV_64FC1);
            rot.copyTo(Rt1.rowRange(0,3).colRange(0,3));
            trans.copyTo(Rt1.rowRange(0,3).col(3));

            cv::Mat point3d_homo0;
            cv::triangulatePoints(Rt0, Rt1,
                                          triangulation_points1, triangulation_points2,
                                          point3d_homo0);
            assert(point3d_homo0.cols == triangulation_points1.size());

            double dist = 50.0;
            cv::Mat mask_trg = point3d_homo0.row(2).mul(point3d_homo0.row(3)) > 0;
            point3d_homo0.row(0) /= point3d_homo0.row(3);
            point3d_homo0.row(1) /= point3d_homo0.row(3);
            point3d_homo0.row(2) /= point3d_homo0.row(3);
            point3d_homo0.row(3) /= point3d_homo0.row(3);

            mask_trg = (point3d_homo0.row(2) < dist) & mask_trg;
            cv::Mat point3d_homo1 = Rt1*point3d_homo0;
            mask_trg = (point3d_homo1.row(2) > 0) & mask_trg;
            mask_trg = (point3d_homo1.row(2) < dist) & mask_trg;

            int goodPoints = countNonZero(mask_trg);

            std::vector<cv::Point2d> good_features_1_uv, good_features_2_uv;
            if((double)goodPoints > 0.75*(double)inlier_cnt) {
                Eigen::Matrix3d Rotation;
                Eigen::Vector3d Translation;
                Eigen::Matrix3d R;
                Eigen::Vector3d T;
                for (int i = 0; i < 3; i++) {
                    T(i) = trans.at<double>(i, 0);
                    for (int j = 0; j < 3; j++)
                        R(i, j) = rot.at<double>(i, j);
                }

                Rotation = R.transpose();
                Translation = -R.transpose() * T;
                w_T_i.block<3, 3>(0, 0) = Rotation;
                w_T_i.block<3, 1>(0, 3) = Translation;
                ROS_WARN_STREAM("Frame by  Frame Trans: \n" << Rotation.eulerAngles(0, 1, 2).transpose()*180/M_PI << " " << Translation.transpose());
                viewTrajectory(w_T_i.block(0, 3, 3, 1));
                std::vector<cv::Point3d> points3d0_homo_good(goodPoints);
                std::vector<cv::Point3d> points3d1_homo_good(goodPoints);
                for(int i = 0; i < mask_trg.cols; i++) {
                    if(mask_trg.at<unsigned char>(i)){
                        cv::Point3d pt0;
                        pt0.x = point3d_homo0.at<double>(0, i);
                        pt0.y = point3d_homo0.at<double>(1, i);
                        pt0.z = point3d_homo0.at<double>(2, i);
                        points3d0_homo_good.push_back(pt0);

                        cv::Point3d pt1;
                        pt1.x = point3d_homo1.at<double>(0, i);
                        pt1.y = point3d_homo1.at<double>(1, i);
                        pt1.z = point3d_homo1.at<double>(2, i);
                        points3d1_homo_good.push_back(pt1);

                        good_features_1_uv.push_back(triangulation_points1_uv[i]);
                        good_features_2_uv.push_back(triangulation_points2_uv[i]);

                        mapPoint mp_id;
                        mp_id.pointXYZ = cv::Point3d(pt0.x, pt0.y, pt0.z);
                        mp_id.id = triangulation_ids[i];
                        mp_id.point_uv = triangulation_points1_uv[i];
                        mp_id.point_xy = triangulation_points1[i];
                        mp_id.T_obspose = Eigen::Matrix4d::Identity();
                        mp_id.ref_frame = 0;
                        map_points_id.push_back(mp_id);
                    }
                }
                if(view_initialization) {
                    viewInitialization(points3d0_homo_good,
                                           rot, trans,
                                           good_features_1_uv, good_features_2_uv);
                }
                initialized = true;
                ROS_WARN_STREAM("Initialized!! at frame: " << frame_no << " With " << map_points_id.size() << " points");
                std::cout << std::endl;
            } else {
                ROS_WARN_STREAM("Not enough points after triangulation. check 'dist'");
            }
        } else {
            ROS_WARN_STREAM("Corresponding matches very close to each other in image plane");
        }
    }

    double getRelativeScale(cv::Mat rvec, cv::Mat tvec) {
//        std::vector<int> common_indices =
//                getMatchIds(prev_idx, curr_idx);
        std::vector<cv::Point2d> prev_pts;
        std::vector<cv::Point2d> curr_pts;
        // Find common ids
        for(int count = 0; count < common_ids.size(); count++) {
            int query_id = common_ids[count];
            for(int i = 0; i < prev_features.size(); i++) {
                if(query_id == prev_features[i].id){
                    prev_pts.push_back(prev_features[i].point_xy);
                    break;
                }
            }
            for(int i = 0; i < curr_features.size(); i++) {
                if(query_id == curr_features[i].id){
                    curr_pts.push_back(curr_features[i].point_xy);
                    break;
                }
            }
        }

        cv::Mat R;
        cv::Rodrigues(rvec, R);
        cv::Mat Rt0 = cv::Mat::eye(3, 4, CV_64FC1);
        cv::Mat Rt1 = cv::Mat::eye(3, 4, CV_64FC1);
        R.copyTo(Rt1.rowRange(0,3).colRange(0,3));
        tvec.copyTo(Rt1.rowRange(0,3).col(3));

        cv::Mat point3d_homo0;
        cv::Mat point3d_homo1;
        cv::triangulatePoints(Rt0, Rt1,
                              prev_pts, curr_pts,
                              point3d_homo0);
        assert(point3d_homo0.cols == prev_pts.size());

        double dist = 50.0;
        cv::Mat mask_trg = point3d_homo0.row(2).mul(point3d_homo0.row(3)) > 0;
        point3d_homo0.row(0) /= point3d_homo0.row(3);
        point3d_homo0.row(1) /= point3d_homo0.row(3);
        point3d_homo0.row(2) /= point3d_homo0.row(3);
        point3d_homo0.row(3) /= point3d_homo0.row(3);

        mask_trg = (point3d_homo0.row(2) < dist) & mask_trg;
        point3d_homo1 = Rt1*point3d_homo0;
        mask_trg = (point3d_homo1.row(2) > 0) & mask_trg;
        mask_trg = (point3d_homo1.row(2) < dist) & mask_trg;

        int goodPoints = countNonZero(mask_trg);
        _3dPoints_k.clear();
        for(int i = 0; i < mask_trg.cols; i++) {
            if(mask_trg.at<unsigned char>(i)){
                cv::Point3d pt1;
                pt1.x = point3d_homo1.at<double>(0, i);
                pt1.y = point3d_homo1.at<double>(1, i);
                pt1.z = point3d_homo1.at<double>(2, i);
                point3DWiD pt3d_id;
                pt3d_id.point_3d = cv::Point3d(pt1);
                pt3d_id.id = common_ids[i];
                _3dPoints_k.push_back(pt3d_id);
            }
        }

        std::vector<cv::Point3d> X_k_1;
        std::vector<cv::Point3d> X_k;
        for(int i = 0; i < _3dPoints_k_1.size(); i++) {
            for(int j = 0; j < _3dPoints_k.size(); j++) {
                if(_3dPoints_k_1[i].id == _3dPoints_k[j].id) {
                    X_k_1.push_back(_3dPoints_k_1[i].point_3d);
                    X_k.push_back(_3dPoints_k[j].point_3d);
                    break;
                }
            }
        }
        assert(X_k.size() == X_k_1.size());
        ROS_INFO_STREAM("No of matches: " << X_k_1.size());
        int n = X_k.size();
        double count = 0;
        double sum = 0;
        double threshold = 0.00001;
        for(int i = 0; i < n-1; i++) {
            for(int j = i+1; j < n; j++) {
                double dist_k = cv::norm(X_k[i] - X_k[j]);
                double dist_k_1 = cv::norm(X_k_1[i] - X_k_1[j]);
                if(dist_k < threshold)
                    continue;
                count += 1;
                sum +=  dist_k_1/dist_k;
            }
        }
        ROS_INFO_STREAM("Count = " << count);
        sum /= count;
        if(isnan(sum) || isinf(sum))
            sum = 1;
        return sum;
    }

    void recoverPosebyPnP() {
        ROS_WARN_STREAM("Current Frame No: " << frame_no);
        ROS_WARN_STREAM("No of map points: " << map_points_id.size());
        ROS_WARN_STREAM("No of points waiting: " << triangulation_candidates.size());
        cv::Mat rvec, tvec;
        // Match 2D-3D correspondences
        std::vector<cv::Point2d> features_2d;
        std::vector<cv::Point2d> features_2d_uv;
        std::vector<cv::Point3d> features_3d;
        std::vector<int> match_indices;
        std::vector<int> remove_indices;

        _3dPoints_k_1.clear();
        for(int i = 0; i < map_points_id.size(); i++) {
            bool found_match = false;
            for(int j = 0; j < curr_features.size(); j++) {
                if(curr_features[j].id == map_points_id[i].id) {
                    features_2d.push_back(curr_features[j].point_xy);
                    features_2d_uv.push_back(curr_features[j].point_uv);
                    match_indices.push_back(curr_features[j].id);

                    cv::Point3d X_obs = map_points_id[i].pointXYZ;
                    Eigen::Vector4d X_obs_eig = Eigen::Vector4d(X_obs.x, X_obs.y, X_obs.z, 1);
                    Eigen::Matrix4d w_T_obs = map_points_id[i].T_obspose;
                    Eigen::Matrix4d w_T_curr_1 = w_T_i;
                    Eigen::Vector4d X_curr_1_eig = w_T_curr_1.inverse()*w_T_obs*X_obs_eig;

                    features_3d.push_back(cv::Point3d(X_curr_1_eig.x(), X_curr_1_eig.y(), X_curr_1_eig.z()));
                    point3DWiD pt3d_id;
                    pt3d_id.point_3d = cv::Point3d(X_curr_1_eig.x(), X_curr_1_eig.y(), X_curr_1_eig.z());
                    pt3d_id.id = curr_features[j].id;
                    _3dPoints_k_1.push_back(pt3d_id);
                    found_match = true;
                    break;
                }
            }
            if (!found_match) {
                remove_indices.push_back(i);
            }
        }

        assert(features_2d.size() == features_3d.size());
        assert(remove_indices.size() + features_2d.size() == map_points_id.size());
        assert(match_indices.size() == features_3d.size());

        // Remove Map Points which do not match
        for(int i = 0; i < remove_indices.size(); i++)
            map_points_id.erase(map_points_id.begin()+remove_indices[i]);

        int no_of_3d_2d_matches = features_2d.size();

        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat D = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);

        cv::Mat inliers;
        std::vector<uchar> status;

        int inlier_count = 0;
        std::vector<cv::Point3d> feature_3d_outlier_free;
        std::vector<cv::Point2d> feature_2d_outlier_free;
        std::vector<cv::Point2d> feature_2d_uv_outlier_free;
        std::vector<int> match_indices_outlier_free;
        remove_indices.clear();
        ROS_WARN_STREAM("No of points used in SolvePnP: " << features_3d.size());
        cv::solvePnPRansac(features_3d, features_2d, K, D, rvec, tvec,
                false, 100,
                           rep_err / 460.0,
                           0.99, inliers);
//        relative_scale = getRelativeScale(rvec, tvec);
//        ROS_INFO_STREAM("Relative Scale: " << getRelativeScale(rvec, tvec));

        for (int i = 0; i < (int)features_2d.size(); i++)
            status.push_back(0);

        for( int i = 0; i < inliers.rows; i++) {
            int n = inliers.at<int>(i);
            status[n] = 1;
        }
//        if(status.size() == 0)
//            ROS_ERROR_STREAM("No Inliers for SolvePnP");
        for( int i = 0; i < status.size(); i++) {
            if((int)status[i] != 0) {
                inlier_count++;
                feature_3d_outlier_free.push_back(features_3d[i]);
                feature_2d_outlier_free.push_back(features_2d[i]);
                feature_2d_uv_outlier_free.push_back(features_2d_uv[i]);
                match_indices_outlier_free.push_back(match_indices[i]);
            } else {
                remove_indices.push_back(i);
            }
        }

        assert(remove_indices.size()+inlier_count == features_3d.size());
        // Remove Map Points which prove to be outliers
        for(int i = 0; i < remove_indices.size(); i++)
            map_points_id.erase(map_points_id.begin()+remove_indices[i]);

        std::vector<cv::Point2d> projected_features;
        cv::projectPoints(feature_3d_outlier_free,
                rvec, relative_scale*tvec, K_, D_,
                projected_features, cv::noArray(), 0);


        for (int i = 0; i < feature_2d_uv_outlier_free.size(); i++) {
            cv::circle(curr_image, feature_2d_uv_outlier_free[i], 2,
                       cv::Scalar(0, 255,  0), 10);
            cv::circle(curr_image, projected_features[i], 2,
                       cv::Scalar(0, 0,  255), 10);
            char name[10];
            sprintf(name, "%d", match_indices_outlier_free[i]);
            cv::putText(curr_image, name, feature_2d_uv_outlier_free[i],
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 255, 0));
            cv::putText(curr_image, name, projected_features[i],
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 0, 255));
            cv::line(curr_image, feature_2d_uv_outlier_free[i],
                     projected_features[i], cv::Scalar(255,0,0), 2);
        }

        assert(projected_features.size() == feature_3d_outlier_free.size());
        assert(feature_2d_uv_outlier_free.size() == feature_3d_outlier_free.size());

        double sum_rep_err = 0;
        for(int i = 0; i < feature_2d_uv_outlier_free.size(); i++) {
            cv::Point2d x1 = feature_2d_uv_outlier_free[i];
            cv::Point2d x2 = projected_features[i];
            cv::Point2d diff = x1-x2;
            sum_rep_err += cv::norm(diff);
        }
        sum_rep_err  /= feature_2d_uv_outlier_free.size();

        ROS_WARN_STREAM("% of inliers post PnP: " << 100.0f*(double)inlier_count/(double)features_3d.size());
        ROS_WARN_STREAM("Mean Rep Err= " << sum_rep_err);

        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        for (int i = 0; i < 3; i++) {
            T(i) = tvec.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = R_cv.at<double>(i, j);
        }

        Eigen::Matrix3d Rotation;
        Eigen::Vector3d Translation;
        Rotation = R.transpose();
        Translation = -R.transpose() * T;
        Eigen::Matrix4d T_ = Eigen::Matrix4d::Identity();
        T_.block<3, 3>(0, 0) = Rotation;
        T_.block<3, 1>(0, 3) = relative_scale*Translation;
        w_T_i = w_T_i * T_;
        Eigen::Matrix3d Rotn_ = w_T_i.block(0, 0, 3, 3);
        Eigen::Vector3d Trans_ = w_T_i.block(0, 3, 3, 1);
        ROS_WARN_STREAM("Frame by  Frame Trans: \n" << Rotation.eulerAngles(0, 1, 2).transpose()*180/M_PI << " " << Translation.transpose());
        viewTrajectory(w_T_i.block(0, 3, 3, 1));
        cv::resize(curr_image, curr_image,
                       cv::Size(0, 0), 1, 1);

        if(sensor_name == "pylon")
            resize(curr_image, curr_image, cv::Size(0, 0), 0.5, 0.5);
        cv::imshow("Visual Odometry Tracking", curr_image);
        cv::waitKey(1);

        triangulateNewLandMarks(R_cv, tvec);
        std::cout << std::endl;
    }

    void addLandmarksToWaitList(std::vector<int> ids) {
        for(int i = 0; i < ids.size(); i++) {
            int id = ids[i];
            for (int j = 0; j < curr_features.size(); j++) {
                if(id == curr_features[j].id) {
                    feature_id_pose fIdPose;
                    fIdPose.id = id;
                    fIdPose.pt_uv = curr_features[j].point_uv;
                    fIdPose.pt_xy = curr_features[j].point_xy;
                    fIdPose.obs_pose = w_T_i;
                    fIdPose.image_frame_no = frame_no;
                    triangulation_candidates.push_back(fIdPose);
                    break;
                }
            }
        }
    }

    std::pair<std::vector<int>, std::vector<int> >  separateWaitngAndNonWaitingIds(std::vector<int> unmapped_ids) {
        std::vector<int> waiting_ids;
        std::vector<int> nonwaiting_ids;
        for(int i = 0; i < unmapped_ids.size(); i++) {
            bool found = false;
            for(int j = 0; j < triangulation_candidates.size(); j++) {
                if(unmapped_ids[i] == triangulation_candidates[j].id) {
                    waiting_ids.push_back(unmapped_ids[i]);
                    found = true;
                    break;
                }
            }
            if(!found) {
                nonwaiting_ids.push_back(unmapped_ids[i]);
            }
        }
        assert(waiting_ids.size()+nonwaiting_ids.size()==unmapped_ids.size());
        std::pair<std::vector<int>, std::vector<int> > waiting_nonwaiting_ids;
        waiting_nonwaiting_ids.first = waiting_ids;
        waiting_nonwaiting_ids.second = nonwaiting_ids;
        return waiting_nonwaiting_ids;
    }

    void addLandmarksToMap() {
        // First we have to check if the triangulation gives suff parallex
        int no_of_candidates_waiting = triangulation_candidates.size();
        int no_of_curr_features = curr_features.size();
        std::vector<std::pair<int, int> > waitIndices_currIndices;
        for(int i = 0; i < no_of_candidates_waiting; i++) {
            int query_id = triangulation_candidates[i].id;
            for(int j = 0; j < no_of_curr_features; j++) {
                int database_id = curr_features[j].id;
                if(query_id == database_id) {
                    std::pair<int, int> waitIndex_currIndex;
                    waitIndex_currIndex.first = i;
                    waitIndex_currIndex.second = j;
                    waitIndices_currIndices.push_back(waitIndex_currIndex);
                    break;
                }
            }
        }
        std::vector<int> remove_waiting_indices;
        double avg_angle_value = 0;
        for(int i = 0; i < waitIndices_currIndices.size(); i++) {
            int wait_index = waitIndices_currIndices[i].first;
            int curr_index = waitIndices_currIndices[i].second;

            feature_id_pose fIDPose = triangulation_candidates[wait_index];
            pointWiD curr_feature = curr_features[curr_index];

            assert(fIDPose.id == curr_feature.id);

            if(fIDPose.image_frame_no != curr_feature.ref_frame) {
                cv::Point2d waiting_xy =fIDPose.pt_xy;
                std::vector<cv::Point2d> waiting_xy_vec;
                waiting_xy_vec.push_back(waiting_xy);
                Eigen::Matrix4d w_T_waiting = fIDPose.obs_pose;

                cv::Point2d curr_xy = curr_feature.point_xy;
                std::vector<cv::Point2d> curr_xy_vec;
                curr_xy_vec.push_back(curr_xy);
                Eigen::Matrix4d w_T_curr = w_T_i;

                Eigen::Matrix4d curr_T_wait = w_T_curr.inverse() * w_T_waiting;
                Eigen::Matrix4d wait_T_curr = curr_T_wait.inverse();
                Eigen::Matrix3d curr_R_wait_eig = curr_T_wait.block(0, 0, 3, 3);
                Eigen::Vector3d curr_t_wait_eig = curr_T_wait.block(0, 3, 3, 1);
                cv::Mat curr_R_wait_cv;
                cv::Mat curr_t_wait_cv;
                cv::eigen2cv(curr_R_wait_eig, curr_R_wait_cv);
                cv::eigen2cv(curr_t_wait_eig, curr_t_wait_cv);

                cv::Mat Rt0 = cv::Mat::eye(3, 4, CV_64FC1);
                cv::Mat Rt1 = cv::Mat::eye(3, 4, CV_64FC1);
                curr_R_wait_cv.copyTo(Rt1.rowRange(0,3).colRange(0,3));
                curr_t_wait_cv.copyTo(Rt1.rowRange(0,3).col(3));

                cv::Mat point3d_homo0;
                cv::Mat point3d_homo1;
                cv::triangulatePoints(Rt0, Rt1,
                                      waiting_xy_vec, curr_xy_vec,
                                      point3d_homo0);
                assert(point3d_homo0.cols == 1);

                double dist = max_dist;
                cv::Mat mask_trg =
                        point3d_homo0.row(2).mul(point3d_homo0.row(3)) > 0;
                point3d_homo0.row(0) /= point3d_homo0.row(3);
                point3d_homo0.row(1) /= point3d_homo0.row(3);
                point3d_homo0.row(2) /= point3d_homo0.row(3);
                point3d_homo0.row(3) /= point3d_homo0.row(3);

                mask_trg = (point3d_homo0.row(2) < dist) & mask_trg;
                point3d_homo1 = Rt1*point3d_homo0;
                mask_trg = (point3d_homo1.row(2) > 0) & mask_trg;
                mask_trg = (point3d_homo1.row(2) < dist) & mask_trg;

                int goodPoints = countNonZero(mask_trg);
                if(goodPoints == 1) {
                    Eigen::Vector3d ptXYZ_wait = Eigen::Vector3d(point3d_homo0.at<double>(0),
                                                                 point3d_homo0.at<double>(1),
                                                                 point3d_homo0.at<double>(2));
                    Eigen::Vector3d ptXYZ_curr = Eigen::Vector3d(point3d_homo1.at<double>(0),
                                                                 point3d_homo1.at<double>(1),
                                                                 point3d_homo1.at<double>(2));
                    Eigen::Vector3d vec_0 = -ptXYZ_wait;
                    Eigen::Vector3d vec_1 =
                            wait_T_curr.block(0, 3, 3, 1)
                            - ptXYZ_wait;

                    double cosAngle = vec_0.dot(vec_1)/(vec_0.norm()*vec_1.norm());
                    double angle = acos(cosAngle);
                    angle = atan2(sin(angle), cos(angle));
                    angle = angle*180/M_PI;
                    avg_angle_value += fabs(angle);
//                    ROS_WARN_STREAM("Angle: " << angle);
                    if (fabs(angle) > 0.1) {
                        mapPoint mp_id;
                        mp_id.pointXYZ = cv::Point3d(ptXYZ_wait.x(), ptXYZ_wait.y(), ptXYZ_wait.z());
                        mp_id.id = fIDPose.id;
                        mp_id.T_obspose = fIDPose.obs_pose;
                        mp_id.point_xy = fIDPose.pt_xy;
                        mp_id.point_uv = fIDPose.pt_uv;
                        mp_id.ref_frame = fIDPose.image_frame_no;
                        map_points_id.push_back(mp_id);
                        remove_waiting_indices.push_back(wait_index);
                    }
                }
            }
        }
//        ROS_WARN_STREAM("Avg parallex angle: " << avg_angle_value/(double)remove_waiting_indices.size());
        for(int i = 0; i < remove_waiting_indices.size(); i++)
            triangulation_candidates.erase(triangulation_candidates.begin()+remove_waiting_indices[i]);
        remove_waiting_indices.clear();

        for(int i = 0; i < triangulation_candidates.size(); i++){
            if(fabs(frame_no-triangulation_candidates[i].image_frame_no) > 5)
                remove_waiting_indices.push_back(i);
        }
        for(int i = 0; i < remove_waiting_indices.size(); i++)
            triangulation_candidates.erase(triangulation_candidates.begin()+remove_waiting_indices[i]);
    }

    void triangulateNewLandMarks(cv::Mat R, cv::Mat t) {
//        std::vector<int> common_ids =
//                getMatchIds(prev_idx, curr_idx);
        //Only keep ids which are not in the mapped list
        std::vector<int> unmapped_ids;
        for(int i = 0; i < common_ids.size(); i++) {
            int query_id = common_ids[i];
            bool found_id = false;
            for (int j = 0; j < map_points_id.size(); j++) {
                if(query_id == map_points_id[j].id) {
                    found_id = true;
                    break;
                }
            }
            if (!found_id)
                unmapped_ids.push_back(query_id);
        }

        int no_of_new_unmapped_pts = unmapped_ids.size();
        int no_of_candidates_waiting = triangulation_candidates.size();
        if(no_of_candidates_waiting == 0) {
            // This will happen in the first pass
            addLandmarksToWaitList(unmapped_ids);
        } else {
            // Check if candidates in the waiting list could be mapped
            // Those. that can be mapped must be added to the mapped list and removed from waiting list
            addLandmarksToMap();
            // unmapped_ids can be divided into waiting)ids and non_waiting_ids
            // non_waiting_ids must be added to the waiting list
            std::pair<std::vector<int>, std::vector<int> > waiting_nonwaiting_ids =
                    separateWaitngAndNonWaitingIds(unmapped_ids);
            std::vector<int> waiting_ids = waiting_nonwaiting_ids.first;
            std::vector<int> non_waiting_ids = waiting_nonwaiting_ids.second;
            // non_waiting ids must be added to waiting list(triangulation_candidates)
            if(!non_waiting_ids.empty())
                addLandmarksToWaitList(non_waiting_ids);
        }
    }

    void track() {
        common_ids = getMatchIds(prev_idx, curr_idx);
        recoverPosebyPnP();
    }

    void callback(const sensor_msgs::ImageConstPtr &image_msg,
                  const sensor_msgs::PointCloudConstPtr &cloud_msg) {
        curr_image.release();
        curr_features.clear();
        curr_idx.clear();

        try {
            curr_image = cv_bridge::toCvShare(image_msg, "bgr8")->image;
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("Could not convert from '%s' to 'bgr8'.", image_msg->encoding.c_str());
        }

        sensor_msgs::PointCloud feature_pts = *cloud_msg;
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        id_of_point = feature_pts.channels[0];
        u_of_point = feature_pts.channels[1];
        v_of_point = feature_pts.channels[2];
        for (int i = 0; i < id_of_point.values.size(); i++) {
            cv::Point2d pt_xy = cv::Point2d(feature_pts.points[i].x,
                                            feature_pts.points[i].y);
            cv::Point2d pt_uv = cv::Point2d(u_of_point.values[i],
                                            v_of_point.values[i]);
            int idx = id_of_point.values[i];
            pointWiD point_ID;
            point_ID.point_xy = pt_xy;
            point_ID.point_uv = pt_uv;
            point_ID.id = idx;
            point_ID.ref_frame = frame_no;
            curr_features.push_back(point_ID);
            curr_idx.push_back(idx);
        }

        if(!first_frame) {
            if(!initialized) {
                initialize();
            } else {
                track();
            }
        } else {
            init_image = curr_image;
            init_features = curr_features;
            init_idx  = curr_idx;
        }

        prev_features.clear();
        prev_idx.clear();
        prev_features = curr_features;
        prev_idx = curr_idx;

        first_frame = false;
        frame_no++;
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "odometry_node");
    ros::NodeHandle nh("~");
    Odometry odom(nh);
    ros::spin();
    return 0;
}