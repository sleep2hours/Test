#include <iostream>
#include <chrono>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "../include/CameraCtl.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;
int main()
{
    Mat frame;
    cm::CameraCtl cap;
    const cv::Mat NEW_INF_INTRINSIC = (cv::Mat_<double>(3, 3) << 1631.285522460938, 0, 722.7377887028852,
                                       0, 1692.166015625, 541.7929725137074,
                                       0, 0, 1);
    const cv::Mat INF_INTRINSIC = (cv::Mat_<double>(3, 3) << 1770.89047225438, 0, 720,
                                   0, 1769.66942390649, 540,
                                   0, 0, 1);
    const std::vector<float> INF_DIST = std::vector<float>{
        -0.457561248586060,
        0.272138858774427,
        0.00303236038540222,
        0.00225295567632989};
    // cv::VideoCapture;
    int read = cap.startGrabbing();
    if (read)
    {
        std::cout << "Read video failed!" << std::endl;
        return 1;
    }
    namedWindow("output", CV_WINDOW_AUTOSIZE);
    frame = cap.getOpencvMat();
    Size size = Size(frame.cols, frame.rows);
    std::cout << size << std::endl;
    VideoWriter writer;
    writer.open("/home/lym/b22.avi", CV_FOURCC('M', 'J', 'P', 'G'), 50, size, true);
    double _start = std::chrono::system_clock::now().time_since_epoch().count();
    double cnt = 0;
    int flag = 0;
    while (1)
    {
        cnt++;
        frame = cap.getOpencvMat();
        if (waitKey(1) == 'e')
        {
            flag = 1;
        }
        if (flag == 1)
        {
            writer << frame;
            circle(frame, Point(50, 50), 5, Scalar(0, 0, 255), -1);
        }
        imshow("output", frame);
        cv::Mat src;
        undistort(frame, src, INF_INTRINSIC, INF_DIST, NEW_INF_INTRINSIC);
        imshow("undisort", src);
        if (waitKey(1) == 'q')
        {
            break;
        }
    }
    double _end = std::chrono::system_clock::now().time_since_epoch().count();
    double fps = (_end - _start) / cnt / 1e6;
    std::cout << fps << std::endl;
    cap.stopGrabbing();
    writer.release();
    return 0;
}
// int main()
// {
//     Mat img;
//     VideoCapture cap;
//     int heightCamera = 720;
//     int widthCamera = 1280;

//     // Start video capture port 0
//     cap.open(5);

//     // Check if we succeeded
//     if (!cap.isOpened())
//     {
//         cout << "Unable to open camera" << endl;
//         return -1;
//     }
//     // Set frame width and height
//     cap.set(CV_CAP_PROP_FRAME_WIDTH, widthCamera);
//     cap.set(CV_CAP_PROP_FRAME_HEIGHT, heightCamera);
//     cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('X', '2', '6', '4'));

//     // Set camera FPS
//     cap.set(CV_CAP_PROP_FPS, 30);

//     while (true)
//     {
//         // Copy the current frame to an image
//         cap >> img;

//         // Show video streams
//         imshow("Video stream", img);

//         waitKey(1);
//     }

//     // Release video stream
//     cap.release();

//     return 0;
// }
