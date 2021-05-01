#include "../include/CameraCtl.hpp"

namespace cm{
bool CameraCtl::printDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo)
{
    if (NULL == pstMVDevInfo){
        printf("%s\n" , "The Pointer of pstMVDevInfoList is NULL!");
        return false;
    }
    if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE){               //网络设备调用
		// 打印当前相机ip和用户自定义名字
		// print current ip and user defined name
        printf("%s %x\n" , "nCurrentIp:" , pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp);                   //当前IP
        printf("%s %s\n" , "chUserDefinedName:" , pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);     //用户定义名
    }
    else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE){           //usb设备调用
        printf("UserDefinedName:%s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName);
    }
    else{
        printf("Not supported.\n");
    }
    return true;
}

CameraCtl::CameraCtl() {
    _high_exp = true;
    nRet = MV_OK;
    handle = NULL;
    pData = new unsigned char [MAX_IMAGE_DATA_SIZE];
    grabing = false;
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

    // 枚举设备
	// enum device
    nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
    if (MV_OK != nRet) {
        printf("MV_CC_EnumDevices failed! nRet [%x]\n", nRet);
        return;
    }
    if (stDeviceList.nDeviceNum > 0) {
        for (unsigned int i = 0; i < stDeviceList.nDeviceNum; ++i) {
            printf("[device %d]:\n", i);
            MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
            if (NULL == pDeviceInfo) {
                break;
            }
            printDeviceInfo(pDeviceInfo);
        }
    } 
    else {
        printf("Find No Device!\n");
        return;
    }
    // 选择设备并创建句柄
	// select device and create handle
    nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[0]);
    if (MV_OK != nRet) {
        printf("MV_CC_CreateHandle failed! nRet [%x]\n", nRet);
        return;
    }

    // 打开设备
	// open device
    nRet = MV_CC_OpenDevice(handle);
    if (MV_OK != nRet){
        printf("MV_CC_OpenDevice failed! nRet [%x]\n", nRet);
        return;
    }
    nRet = MV_CC_SetHeight(handle, Default_Param.Height);
    nRet = MV_CC_SetWidth(handle, Default_Param.Width);
    if (MV_OK!=nRet)
    {
        printf("MV_CC_SetSize failed! nRet [%x]\n", nRet);
        return;
    }

    //设置帧率
    nRet = MV_CC_SetFrameRate(handle, Default_Param.Fps);
    if (MV_OK != nRet)
    {
        printf("MV_CC_SetFrameRate failed! nRet [%x]\n", nRet);
        return;
    }

    //设置曝光时间
    nRet = MV_CC_SetExposureTime(handle, Default_Param.Exposure_Time);
    if (MV_OK != nRet)
    {
        printf("MV_CC_SetExposureTime failed! nRet [%x]\n", nRet);
        return;
    }

    //设置增益模式,使用固定值增益，不进行自动增益
    //nRet = MV_CC_SetGainMode(handle, Default_Param.Gain_Mode);
    //if (MV_OK != nRet)
    //{
    //    printf("MV_CC_SetGainmode failed! nRet [%x]\n", nRet);
    //    return;
    //}

    //设置增益固定值
    nRet = MV_CC_SetGain(handle, Default_Param.Gain);
    if (MV_OK != nRet)
    {
        printf("MV_CC_SetGain failed! nRet [%x]\n", nRet);
        return;
    }
    
    //设置gamma值
    nRet = MV_CC_SetGamma(handle, Default_Param.Gamma);

    nRet=MV_CC_SetBalanceRatioBlue(handle, Default_Param.b_balance);
    nRet=MV_CC_SetBalanceRatioGreen(handle, Default_Param.g_balance);
    nRet=MV_CC_SetBalanceRatioRed(handle, Default_Param.r_balance);
    if(nRet != MV_OK){
        printf("Failed to set balance ratio. nRet [%x]\n", nRet);
        return;
    }

    //设置曝光模式，使用固定曝光时间或是手动调节时间，不使用自动曝光
    //nRet = MV_CC_SetExposureAutoMode(handle, Default_Param.Auto_Exposure_Mode);
    //if (MV_OK != nRet)
    //{
    //   printf("MV_CC_SetExposure failed! nRet [%x]\n", nRet);
    //    return;
    //}

    //枚举设备
    nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
    if (MV_OK != nRet){
        printf("MV_CC_SetTriggerMode failed! nRet [%x]\n", nRet);
        return;
    }
}

CameraCtl::~CameraCtl() {
    if (grabing)
        stopGrabbing();
	if (pData) {
		delete pData;	
		pData = NULL;
	}
    // 销毁句柄
	// destroy handle
    nRet = MV_CC_DestroyHandle(handle);
    if (MV_OK != nRet){
        printf("MV_CC_DestroyHandle fail! nRet [%x]\n", nRet);
    }
}

int CameraCtl::startGrabbing() {
    // 开始取流
	// start grab image
    nRet = MV_CC_StartGrabbing(handle);
    if (MV_OK != nRet){
        printf("MV_CC_StartGrabbing fail! nRet [%x]\n", nRet);
        return -1;
    }
    grabing = true;
    return 0;
}

int CameraCtl::stopGrabbing() {
    // 停止取流
	// end grab image
    nRet = MV_CC_StopGrabbing(handle);
    if (MV_OK != nRet) {
        printf("MV_CC_StopGrabbing failed! nRet [%x]\n", nRet);
        return -1;
    }
    grabing = false;
    return 0;
}

int CameraCtl::setExposureTime(float t){             //设置曝光时间
    nRet=MV_CC_SetExposureTime(handle, t);
    if(nRet != MV_OK){
        if(t > HIGH_EXP_THRESH) this->_high_exp = true;                 //大于3000us就认为是高曝光
        else this->_high_exp = false;                                   //低曝光
        printf("Failed to set exposure time. nRet [%x]\n", nRet);
        return -1;
    }
    return 0;
}

int CameraCtl::setWhiteBalance(int b, int g, int r){
    nRet=MV_CC_SetBalanceRatioBlue(handle, b);
    nRet=MV_CC_SetBalanceRatioGreen(handle, g);
    nRet=MV_CC_SetBalanceRatioRed(handle, r);
    printf("BGR: %d, %d, %d\n", b, g, r);
    if(nRet != MV_OK){
        printf("Failed to set white balance. nRet [%x]\n", nRet);
        return -1;
    }
    return 0;
}

int CameraCtl::setGain(float gain){
    nRet = MV_CC_SetGain(handle, Default_Param.Gain);
    if (MV_OK != nRet)
    {
        printf("MV_CC_SetGain failed! nRet [%x]\n", nRet);
        return -1;
    }
    return 0;
}

float CameraCtl::getGain(){
    MVCC_FLOATVALUE* val = new MVCC_FLOATVALUE;
    nRet = MV_CC_GetGain(handle, val);
    if(nRet != MV_OK){
        printf("Failed to get gain! nRet [%x]\n", nRet);
        return -1.0;
    }
    float res = val->fCurValue;
    delete val; val = nullptr;
    return res;
}

//此函数用于低性能取图像，帧率上限60fps,只用于简单DEBUG
Mat CameraCtl::getOpencvMat() {
    MV_FRAME_OUT_INFO_EX stImageInfo = {0};
    memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
    unsigned int nDataSize = MAX_IMAGE_DATA_SIZE;

	nRet = MV_CC_GetImageForBGR(handle, pData, nDataSize, &stImageInfo, Default_Param.nMsec);
    Mat img;
	if (nRet == MV_OK) {
		// (stImageInfo.enPixelType == PixelType_Gvsp_BGR8_Packed)
        img = Mat(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, pData);
    }
    else
    {
        printf("MV_CC_GetImage failed!\n");
    }
    return img;

}
}
