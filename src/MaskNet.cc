/**
* This file is part of DynaSLAM.
*
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/

#include "MaskNet.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <dirent.h>
#include <errno.h>

namespace DynaSLAM
{
// 아래의 코드는  timeval이라는 구조체에 현재 시간(초, usec)을 얻어옵니다.
#define U_SEGSt(a)\
    gettimeofday(&tvsv,0);\
    a = tvsv.tv_sec + tvsv.tv_usec/1000000.0
struct timeval tvsv;
double t1sv, t2sv,t0sv,t3sv;
void tic_initsv(){U_SEGSt(t0sv);}//현재 시간을 저장합니다.
void toc_finalsv(double &time){U_SEGSt(t3sv); time =  (t3sv- t0sv)/1;}
void ticsv(){U_SEGSt(t1sv);}
void tocsv(){U_SEGSt(t2sv);}
// std::cout << (t2sv - t1sv)/1 << std::endl;}

SegmentDynObject::SegmentDynObject(){
    std::cout << "Importing Mask R-CNN Settings..." << std::endl;
    ImportSettings();
    std::string x;
    setenv("PYTHONPATH", this->py_path.c_str(), 1);//현재 코드 실행간에 환경변수를 등록합니다.char*의 형태로 저장되며 1을 마지막 인자로 둬서 저장합니다.
    x = getenv("PYTHONPATH");
    Py_Initialize();
    this->cvt = new NDArrayConverter();
    this->py_module = PyImport_ImportModule(this->module_name.c_str());
    assert(this->py_module != NULL);
    this->py_class = PyObject_GetAttrString(this->py_module, this->class_name.c_str());
    assert(this->py_class != NULL);
    this->net = PyInstance_New(this->py_class, NULL, NULL);//새로운 MaskNet 인스턴스를 만듭니다.
    assert(this->net != NULL);
    std::cout << "Creating net instance..." << std::endl;
    cv::Mat image  = cv::Mat::zeros(480,640,CV_8UC3); //Be careful with size!!
    std::cout << "Loading net parameters..." << std::endl;
    cv::Mat return_mask;
    GetSegmentation_label(image,return_mask);//초기화과정으로 데이터의 세그먼트를 얻어오면서 테스트합니다.
}

SegmentDynObject::~SegmentDynObject(){
    delete this->py_module;
    delete this->py_class;
    delete this->net;
    delete this->cvt;
}

cv::Mat SegmentDynObject::GetSegmentation(cv::Mat &image,std::string dir, std::string name){
    cv::Mat seg = cv::imread(dir+"/"+name,CV_LOAD_IMAGE_UNCHANGED);//이미지 경로로부터 데이터를 얻어옵니다.
    if(seg.empty()){//이전에 만들어지지 않았다면
        PyObject* py_image = cvt->toNDArray(image.clone());//이미지를 깊은 복사해서 numpy 객체로 만듭니다.
        assert(py_image != NULL);
        PyObject* py_mask_image = PyObject_CallMethod(this->net, const_cast<char*>(this->get_dyn_seg.c_str()),"(O)",py_image);//마스크 값을 가져옵니다.
        //해당 객체에서 특정함수를 불러옵니다.
        seg = cvt->toMat(py_mask_image).clone();//mat의 형태로 반환합니다.
        seg.cv::Mat::convertTo(seg,CV_8U);//0 background y 1 foreground  #2진으로 받아옵니다.
        if(dir.compare("no_save")!=0){//이게 초기 인자라서 이값이 있는지 확인한다.
            DIR* _dir = opendir(dir.c_str());
            if (_dir) {closedir(_dir);}//경로가있다면  닫는다.
            else if (ENOENT == errno)
            {
                const int check = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);//특정경로를 만든다.
                if (check == -1) {//오류가 발생하는 경우
                    std::string str = dir;
                    str.replace(str.end() - 6, str.end(), "");//경로의 마지막을 수정합니다.
                    mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                }
            }
            cv::imwrite(dir+"/"+name,seg);//segment된 결과를 저장합니다.
        }
    }
    return seg;//불러오거나 반환합니다.
}

cv::Mat SegmentDynObject::GetSegmentation_label(cv::Mat &image,cv::Mat &label,std::string dir, std::string name){
    cv::Mat seg = cv::imread(dir+"/"+name,CV_LOAD_IMAGE_UNCHANGED);//이미지 경로로부터 데이터를 얻어옵니다.
    if(seg.empty()){//이전에 만들어지지 않았다면
        PyObject* py_image = cvt->toNDArray(image.clone());//이미지를 깊은 복사해서 numpy 객체로 만듭니다.
        assert(py_image != NULL);
        PyObject* py_mask_image = PyObject_CallMethod(this->net, const_cast<char*>(this->get_dyn_seg.c_str()),"(O)",py_image);//마스크 값을 가져옵니다.
        //해당 객체에서 특정함수를 불러옵니다.
        seg = cvt->toMat(py_mask_image).clone();//mat의 형태로 반환합니다.
        seg.cv::Mat::convertTo(seg,CV_8U);//0 background y 1 foreground  #2진으로 받아옵니다.
        if(dir.compare("no_save")!=0){//이게 초기 인자라서 이값이 있는지 확인한다.
            DIR* _dir = opendir(dir.c_str());
            if (_dir) {closedir(_dir);}//경로가있다면  닫는다.
            else if (ENOENT == errno)
            {
                const int check = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);//특정경로를 만든다.
                if (check == -1) {//오류가 발생하는 경우
                    std::string str = dir;
                    str.replace(str.end() - 6, str.end(), "");//경로의 마지막을 수정합니다.
                    mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                }
            }
            cv::imwrite(dir+"/"+name,seg);//segment된 결과를 저장합니다.
        }
        std::string func_name("GetOriginMask");
        PyObject* py_label_image = PyObject_CallMethod(this->net, const_cast<char*>(func_name.c_str()),"(O)",py_image);//마스크 값을 가져옵니다.
        cv::Mat seg_value = cvt->toMat(py_label_image).clone();
        seg_value.cv::Mat::convertTo(label,CV_8U);//가져옵니다.
    }
    return seg;//불러오거나 반환합니다.
}

void SegmentDynObject::ImportSettings(){
    std::string strSettingsFile = "./Examples/RGB-D/MaskSettings.yaml";
    std::cout<<strSettingsFile.c_str()<<std::endl;
    cv::FileStorage fs(strSettingsFile.c_str(), cv::FileStorage::READ);//셋팅을 yaml파일에서 얻어옵니다.
    fs["py_path"] >> this->py_path;//파이썬 코드가 존재하는 경로
    fs["module_name"] >> this->module_name;//사용하는 코드 종류
    fs["class_name"] >> this->class_name;// 사용하는 클래스 이름
    fs["get_dyn_seg"] >> this->get_dyn_seg;// 사용하는 함수 이름

    // std::cout << "    py_path: "<< this->py_path << std::endl;
    // std::cout << "    module_name: "<< this->module_name << std::endl;
    // std::cout << "    class_name: "<< this->class_name << std::endl;
    // std::cout << "    get_dyn_seg: "<< this->get_dyn_seg << std::endl;
}


}






















