#ifndef _USE_LK_OPTICAL_FLOW_
#define _USE_LK_OPTICAL_FLOW_

#include <eigen3/Eigen/Eigen>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

void myCalcOpticalFlowLK(cv::Mat image1, cv::Mat image2,
                        std::vector<cv::Point2f> &prev_kp, std::vector<cv::Point2f> &next_kp,
                        std::vector<uchar> &status, bool use_magic_operation=false, int patch_size=21, int iteration=10, 
                        bool do_prediction=false);

void myCalcOpticalFlowPyrLK(cv::Mat image1, cv::Mat image2,
                            std::vector<cv::Point2f> &prev_kp, std::vector<cv::Point2f> &next_kp,
                            std::vector<uchar> &status, int patch_size=21, int iteration=10, int start_level=1);

#endif