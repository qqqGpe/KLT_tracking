#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include "klt.h"

int main( int argc, char** argv )
{
    std::string path_to_dataset = "/home/gao/projects/cpp_projects/LKoptical/data";
    std::string associate_file = path_to_dataset + "/associate.txt";
    std::cout << "associate file is: " << associate_file << std::endl;

    std::ifstream fin( associate_file );
    if ( !fin ) 
    {
        std::cerr<< "I cann't find associate.txt!" << std::endl;
        exit(1);
    }

    std::string rgb_file, depth_file, time_rgb, time_depth;
    std::list< cv::Point2f > keypoints; 
    cv::Mat image, depth, last_image;

    for ( int index=0; index<100; index++ )
    {
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        image = cv::imread( path_to_dataset + "/" + rgb_file );
        if (index ==0 )
        {
            // extract FAST keypoints for the first frame
            std::vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect(image, kps);
            for (auto kp:kps)
                keypoints.push_back(kp.pt);
            last_image = image;
            continue;
        }
        if ( image.data == nullptr)
        {
            continue;
        }

        // tracking with LK 
        std::vector<cv::Point2f> next_keypoints;
        std::vector<cv::Point2f> prev_keypoints;
        std::vector<uchar> status;
        std::vector<float> error;
        for ( auto kp:keypoints )
        {
            prev_keypoints.push_back(kp);
        }
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        myCalcOpticalFlowLK(last_image, image, prev_keypoints, next_keypoints, status, false, 21, 10, false); 
        // myCalcOpticalFlowPyrLK(last_image, image, prev_keypoints, next_keypoints, status, 21, 10, 2);
        // cv::calcOpticalFlowPyrLK( last_image, image, prev_keypoints, next_keypoints, status, error, cv::Size(21, 21), 3); // official implementation
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>( t2 - t1 );
        std::cout<<"LK Flow use time: "<< time_used.count() << " seconds." << std::endl;
        // remove lost points
        int i=0; 
        for ( auto iter = keypoints.begin(); iter != keypoints.end(); i++)
        {
            if ( status[i] == 0 )
            {
                iter = keypoints.erase(iter);
                continue;
            }
            *iter = next_keypoints[i];
            iter++;
        }
        std::cout << "tracked keypoints: " << keypoints.size() << std::endl;
        if (keypoints.size() == 0)
        {
            std::cout << "all keypoints are lost." << std::endl;
            break;
        }

        // show the optical flow
        for (int j = 0; j < prev_keypoints.size(); j++)
        {
            if (status[j])
            {
                cv::line(last_image, prev_keypoints[j], next_keypoints[j], cv::Scalar(255, 0, 0), 2);
                cv::circle(last_image, prev_keypoints[j], 3, cv::Scalar(0, 0, 255), -1);
            }
        }

        // show image
        cv::imshow("Optical Flow", last_image);
        cv::imwrite("/home/gao/projects/cpp_projects/LKoptical/LKFlow/LKoptical.png", last_image);
        cv::waitKey(0);
        last_image = image;
    }
    return 0;
}