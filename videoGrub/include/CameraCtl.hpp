/**
 * file CameraCtl.hpp
 * The CameraCtl Class for using hikvision camera in opencv easily.
 * basics: by Dinger
 * complement:hqy
 * integraion: gqr
 * last change: 2020.1.16 10:07
 * All the params of the camera are integrated into: CameraCtl
 * Except "setExposureTime" which will be used in armor detection.
 */

#ifndef __CAMERA_CTL_HPP
#define __CAMERA_CTL_HPP

#include <stdio.h>
#include <string.h>
#include </opt/MVS/include/MvCameraControl.h>
#include <opencv2/opencv.hpp>
#include "CameraParam.hpp"

using namespace cv;

namespace cm{

#define MAX_IMAGE_DATA_SIZE   (40*1024*1024)
const float HIGH_EXP_THRESH = 3000.0;       //被判断为高曝光的曝光时间阈值

class CameraCtl
{
private:
    int nRet;
    unsigned char * pData;
    bool grabing;
    bool printDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo);
    VideoWriter writer;
    CameraParam Default_Param;
public:
    CameraCtl();
    ~CameraCtl();
    int startGrabbing();
    int stopGrabbing();
    int setExposureTime(float t);
    int setWhiteBalance(int b, int g, int r);              //动态设置白平衡
    int setGain(float gain = 17.0);                   //设置增益
    float getGain();                                        //获取当前增益值
    Mat getOpencvMat();
public:
    bool _high_exp;                                         //高曝光判断标签位
    void* handle;
};
}; // namespace Camera

#endif // __CAMERA_CTL_HPP