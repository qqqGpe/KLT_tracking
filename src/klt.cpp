#include "klt.h"

bool isIndexValid(cv::Mat image, int rowIndex, int colIndex)
{
    if (rowIndex >= 0 && rowIndex < image.rows && colIndex >= 0 && colIndex < image.cols)
    {
        return true;
    }
    return false;
}

void homography_filter(const std::vector<cv::Point2f> &pre_pts, const std::vector<cv::Point2f> &next_pts,
                       const float threshold, std::vector<uchar> &status)
{
    if(status.size() != pre_pts.size())
    {
        std::cout << "status size should be equal with pre_pts" << std::endl;
        return;
    }
    cv::Mat H_mat = cv::findHomography(pre_pts, next_pts, cv::RANSAC, threshold);
    Eigen::MatrixXf H(H_mat.rows, H_mat.cols);
    cv::cv2eigen(H_mat, H);
    for(int i = 0; i < pre_pts.size(); i++)
    {
        Eigen::Vector3f point_src(pre_pts[i].x, pre_pts[i].y, 1.0);
        Eigen::Vector3f point_tgt(next_pts[i].x, next_pts[i].y, 1.0);
        Eigen::Vector3f project_pts;

        project_pts = H * point_src;
        project_pts = project_pts / project_pts(2);
        float dist = (project_pts - point_tgt).norm();

        if(dist < threshold)
        {
            status[i] = 1;
        }
        else
        {
            status[i] = 0;
        }
    }
}

void myCalcOpticalFlowLK(cv::Mat image1, cv::Mat image2,
                        std::vector<cv::Point2f> &prev_kp, std::vector<cv::Point2f> &next_kp,
                        std::vector<uchar> &status, bool magic_operation, int patch_size, int iteration,
                        bool do_prediction)
{
    if(next_kp.empty())
    {
        if(do_prediction)
        {
            std::cerr << "ERROR, next_kp must not be zero!" << std::endl;
            return;
        }
        next_kp.resize(prev_kp.size());
    }
    assert(next_kp.size() == prev_kp.size());

    constexpr float threshold = 0.05;
    cv::Mat image1_gray, image2_gray;
    cv::cvtColor(image1, image1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, image2_gray, cv::COLOR_BGR2GRAY);
    
    for(int i = 0; i < prev_kp.size(); i++)
    {
        Eigen::Vector2f p = Eigen::Vector2f::Zero();
        if(do_prediction)
        {
            p(0) = next_kp[i].x - prev_kp[i].x;
            p(1) = next_kp[i].y - prev_kp[i].y;
        }
        
        int u = prev_kp[i].x;
        int v = prev_kp[i].y;
        Eigen::Vector2f delta_p = Eigen::Vector2f::Zero();
        Eigen::MatrixXf J(2, patch_size * patch_size);
        bool jacob_initialized = false;

        for(int iter = 0; iter <= iteration; iter++)
        {
            Eigen::Matrix2f Hessian = Eigen::Matrix2f::Zero();
            Eigen::Vector2f err_total = Eigen::Vector2f::Zero();
            int pixel_idx_count = -1;
            for(int u_step = -patch_size / 2; u_step < patch_size / 2; u_step++)
            {
                for(int v_step = -patch_size / 2; v_step < patch_size / 2; v_step++)
                {
                    pixel_idx_count++;

                    if(!isIndexValid(image1_gray, v + v_step, u + u_step) || !isIndexValid(image2_gray, v + v_step + p(1), u + u_step + p(0)))
                    {
                        continue;
                    }

                    float err = image1_gray.at<uchar>(v + v_step, u + u_step) - image2_gray.at<uchar>(v + v_step + p(1), u + u_step + p(0));    

                    if(magic_operation)
                    {
                        if(!jacob_initialized)
                        {
                            float jacob_x = 0.5 * ( image2_gray.at<uchar>(v + v_step + p(1), u + u_step + p(0) + 1) - image2_gray.at<uchar>(v + v_step + p(1), u + u_step + p(0) - 1));
                            float jacob_y = 0.5 * ( image2_gray.at<uchar>(v + v_step + p(1) + 1, u + u_step + p(0)) - image2_gray.at<uchar>(v + v_step + p(1) - 1, u + u_step + p(0)));
                            J.block<2, 1>(0, pixel_idx_count) = (Eigen::Vector2f() << jacob_x, jacob_y).finished();
                        }
                    }

                    if(!magic_operation)
                    {
                        float jacob_x = 0.5 * ( image2_gray.at<uchar>(v + v_step + p(1), u + u_step + p(0) + 1) - image2_gray.at<uchar>(v + v_step + p(1), u + u_step + p(0) - 1));
                        float jacob_y = 0.5 * ( image2_gray.at<uchar>(v + v_step + p(1) + 1, u + u_step + p(0)) - image2_gray.at<uchar>(v + v_step + p(1) - 1, u + u_step + p(0)));
                        J.block<2, 1>(0, pixel_idx_count) = (Eigen::Vector2f() << jacob_x, jacob_y).finished();                      
                    }
                    
                    err_total += J.block<2, 1>(0, pixel_idx_count) * err;
                }
            }
            
            if(!magic_operation || !jacob_initialized)
            {
                Hessian = J * J.transpose();
                jacob_initialized = true;
            }

            delta_p = Hessian.ldlt().solve(err_total);
            p += delta_p;

            if(delta_p.norm() < threshold)
            {
                break;
            }
        }
        next_kp[i].x = u + p(0);
        next_kp[i].y = v + p(1);
    }

    status.resize(prev_kp.size());
    homography_filter(prev_kp, next_kp, 3.0, status);
}

void myCalcOpticalFlowPyrLK(cv::Mat image1, cv::Mat image2,
                            std::vector<cv::Point2f> &prev_kp, std::vector<cv::Point2f> &next_kp,
                            std::vector<uchar> &status, int patch_size, int iteration, int start_level)
{
    constexpr float pyr_scale = 0.5;
    std::vector<float> scales = {1.0};
    for(int i = 0; i < start_level - 1; i++)
    {
        scales.push_back(scales.back() * pyr_scale);
    }

    std::vector<cv::Point2f> kp1_pyr;
    std::vector<cv::Point2f> kp2_pyr;
    for(auto kp : prev_kp)
    {
        kp1_pyr.push_back(kp * scales.back());
        kp2_pyr.push_back(kp * scales.back());
    }

    for(int level = start_level - 1; level >= 0; level--)
    {
        cv::Mat image1_pyr;
        cv::Mat image2_pyr;
        cv::resize(image1, image1_pyr, cv::Size(image1.cols * scales[level], image1.rows * scales[level]));
        cv::resize(image2, image2_pyr, cv::Size(image2.cols * scales[level], image2.rows * scales[level]));
        myCalcOpticalFlowLK(image1_pyr, image2_pyr, kp1_pyr, kp2_pyr, status, true, patch_size, iteration, true);

        // std::cout << "pyr scale level: " << level << std::endl; 
        // std::cout << "size of kp1_pyr: " << kp1_pyr.size() << std::endl;
        // std::cout << "size of kp2_pyr: " << kp2_pyr.size() << std::endl;

        if(level == 0)
        {
            break;
        }

        for(int i = 0; i < kp1_pyr.size(); i++)
        {
            kp1_pyr[i] = kp1_pyr[i] / pyr_scale;
            kp2_pyr[i] = kp2_pyr[i] / pyr_scale;
        }
    }

    next_kp = kp2_pyr;
}
