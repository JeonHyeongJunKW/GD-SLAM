/**
* 
* This file is a modified version of Dyna-SLAM.<https://github.com/bertabescos/DynaSLAMs>
*
* This file is part of GD SLAM.
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <unistd.h>
#include<opencv2/core/core.hpp>

#include "Geometry.h"
#include "MaskNet.h"
#include <System.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 5 && argc != 6 && argc != 7)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association (path_to_masks) (path_to_output)" << endl;
        return 1;
    }
    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    std::cout << "nImages: " << nImages << std::endl;

    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Initialize Mask R-CNN
    DynaSLAM::SegmentDynObject *MaskNet;//마스크 넷을 담당하는 객체를 생성합니다.
    if (argc==6 || argc==7)
    {
        cout << "Loading Mask R-CNN. This could take a while..." << endl;
        MaskNet = new DynaSLAM::SegmentDynObject();
        cout << "Mask R-CNN loaded!" << endl;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);//SLAM초기화

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);//이미지 갯수만큼 트랙을 만듭니다.

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Dilation settings
    int dilation_size = 15;
    //dilation 연산을 수행할 커널을 생성합니다.(용도 모름)
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                           cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                           cv::Point( dilation_size, dilation_size ) );
    //#include <sys/stat.h> 
    //#include <sys/types.h> 
    //int mkdir(const char *pathname, mode_t mode);
    //일정 이름으로 경로를ㄹ 만듭니다. 접근권한을 mode로 설정할 수 있습니다.
    /*
    S_IRUSR : (00400) - owner에 대한 읽기 권한 
    S_IWUSR : (00200) - owner에 대한 쓰기 권한 
    S_IXUSR : (00100) - owner에 대한 search 권한 
    S_IRGRP : (00040) - Group에 대한 읽기 권한 
    S_IWGRP : (00020) - Group에 대한 쓰기 권한 
    S_IXGRP : (00010) - Group에 대한 search 권한 
    S_IROTH : (00004) - Other에 대한 읽기 권한 
    S_IWOTH : (00002) - Other에 대한 쓰기 권한 
    S_IXOTH : (00001) - Other에 대한 search 권한

    */
    if (argc==7)
    {
        std::string dir = string(argv[6]);
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);//유저의 읽기 쓰기 실행, 그룹의 읽기 쓰기 실행, 다른사람의 읽기, 다른사람의 사용하기
        dir = string(argv[6]) + "/rgb/";
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        dir = string(argv[6]) + "/depth/";
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        dir = string(argv[6]) + "/mask/";
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }

    // Main loop
        cv::Mat imRGB, imD;
        cv::Mat imRGBOut, imDOut,maskOut;

    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);

        double tframe = vTimestamps[ni];//이미지의 타임스탬프

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Segment out the images
        cv::Mat mask = cv::Mat::ones(480,640,CV_8U);//처음 마스크는 1이다.

        cv::Mat origin_label;
        if (argc == 6 || argc == 7)
        {
            cv::Mat maskRCNN;
            
            maskRCNN = MaskNet->GetSegmentation_label(imRGB,origin_label,string(argv[5]),vstrImageFilenamesRGB[ni].replace(0,4,""));

            // cv::Mat origin_labeldil = maskRCNN.clone();
            // cv::dilate(maskRCNN,maskRCNNdil, kernel);
            // cv::dilate(origin_label,maskRCNNdil, kernel);
            mask = mask - maskRCNN;//dil;//정적인부분이 1로 남는다.
        }

        // Pass the image to the SLAM system(입력인자 6번이 있는경우는 마스크의 출력경로가 있다는 것
        if (argc == 7){
            SLAM.TrackRGBD_GD(imRGB,imD,mask,origin_label,tframe,imRGBOut,imDOut,maskOut);
            // SLAM.TrackRGBD(imRGB,imD,mask,tframe,imRGBOut,imDOut,maskOut);
        }
        else {SLAM.TrackRGBD(imRGB,imD,mask,tframe);}

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        if (argc == 7)
        {
            cv::imwrite(string(argv[6]) + "/rgb/" + vstrImageFilenamesRGB[ni],imRGBOut);
            vstrImageFilenamesD[ni].replace(0,6,"");
            cv::imwrite(string(argv[6]) + "/depth/" + vstrImageFilenamesD[ni],imDOut);//마지막 숫자때고 depth 저장
            cv::imwrite(string(argv[6]) + "/mask/" + vstrImageFilenamesRGB[ni],maskOut);
        }

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();//연산에 걸린시간을 구합니다.(마스킹 등)

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)//현재 프레임이 마지막프레임이 아니라면
            T = vTimestamps[ni+1]-tframe;//다음 프레임의 타임스탬프 차이를 구한다.
        else if(ni>0)//현재프레임이 마지막프레임이라면
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)//프레임 차이보다 크면 그냥 기다린다.
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());//걸린시간을 순서대로 정렬합니다.
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;//중위값
    cout << "mean tracking time: " << totaltime/nImages << endl;//평균값

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");//카메라 경로를 저장합니다.
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");//키프레임 경로를 저장합니다.

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{//association파일을 열어서 시간, RGB파일 경로, Depth파일 경로를 저장합니다.
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}
