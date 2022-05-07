#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <omp.h>
#include "GeoMaskMaker.h"
#include <chrono>
#include <random>
#include <algorithm>
#include <iterator>

using namespace cv;
using namespace std;


string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

GeoMaskMaker::GeoMaskMaker(Mat inst_param, Mat DistCoef, float mDepthMapFactor)
{
    inst_param.copyTo(_inst_param);//내부파라미터를 복사합니다.
    DistCoef.copyTo(_DistCoefParam);//내부파라미터를 복사합니다.
    _DepthMapFactor = mDepthMapFactor;//내부파라미터를 복사합니다.
    _fu = inst_param.at<float>(0,0);
    _fv = inst_param.at<float>(1,1);
    _cu = inst_param.at<float>(0,2);
    _cv = inst_param.at<float>(1,2);
    for(int k=0; k<90; k++)
    {
        
        Vec3b color(rand()%255,rand()%255,rand()%255);
        random_color.push_back(color);
    }
    mimage_height =480;
    mimage_width = 640;
    this->undistortedPoint = Mat(mimage_height*mimage_width,2,CV_32F);
    int idx=0;
    for(int y = 0; y < mimage_height; ++y)
    {
        for(int x = 0; x < mimage_width; ++x)
        {
            undistortedPoint.at<float>(idx,0)=x;
            undistortedPoint.at<float>(idx,1)=y;
            idx++;
        }
    }
    undistortedPoint=undistortedPoint.reshape(2);
    undistortPoints(undistortedPoint,undistortedPoint,_inst_param,_DistCoefParam,cv::Mat(),_inst_param);
    undistortedPoint=undistortedPoint.reshape(1);
}
Point2f GeoMaskMaker::GetUndistortedPixel(int i, int j)
{
    int udp_idx = i*mimage_width+j;
    Point2f returnPoint = Point2f(undistortedPoint.at<float>(udp_idx,0),undistortedPoint.at<float>(udp_idx,1));
    return returnPoint;
}
bool GeoMaskMaker::GetRt(Mat &R, Mat &T)
{
    chrono::system_clock::time_point StartTime = chrono::system_clock::now();
    double scalefactor = 1.2;
    double pyramid_size = 8;
    const static auto& _orb_OrbHandle = ORB::create(2000,scalefactor,pyramid_size,31,0,2);

    Mat des_first;
    vector<KeyPoint>  kp_first;
    _orb_OrbHandle->detectAndCompute(_firstImage,noArray(),kp_first,des_first);

    Mat des_second;
    vector<KeyPoint>  kp_second;
    _orb_OrbHandle->detectAndCompute(_secondImage,noArray(),kp_second,des_second);

    vector<DMatch> matches;
    Ptr<DescriptorMatcher> _match_OrbMatchHandle = BFMatcher::create(NORM_HAMMING,true);
    _match_OrbMatchHandle->match(des_first,des_second,matches);
    sort(matches.begin(),matches.end());
    // cout<<"매칭 수"<<matches.size()<<endl;
    vector<DMatch> good_matchs(matches.begin(), matches.begin()+100);
    vector<Point2f> mp_first;
    vector<Point2f> mp_second;
    for(int i=0; i<good_matchs.size(); i++)
    {
        mp_first.push_back(kp_first[good_matchs[i].queryIdx].pt);
        mp_second.push_back(kp_second[good_matchs[i].trainIdx].pt);
    }
    Mat ud_mat(mp_first.size(),2,CV_32F);
    for(int i=0; i<mp_first.size(); i++)
    {
        ud_mat.at<float>(i,0)=mp_first[i].x;
        ud_mat.at<float>(i,1)=mp_first[i].y;
    }
    ud_mat=ud_mat.reshape(2);
    undistortPoints(ud_mat,ud_mat,_inst_param,_DistCoefParam,cv::Mat(),_inst_param);
    //처음 꺼에 대해서, 왜곡 풀고, 깊이값 확인해서 있는점들만 매칭하면됨
    ud_mat=ud_mat.reshape(1);
    // cout<<ud_mat.rows<<" : "<<ud_mat.cols<<endl;
    vector<Point3f> objectPoints;
    vector<Point2f> imagePixels;

    //depth값 확인하면됨
    Mat meter_depth_mat = _firstDepth.clone();
    Mat inv_inst = _inst_param.inv();
    for(int i=0; i<mp_first.size(); i++)
    {
        int d_x = ud_mat.at<float>(i,0);
        int d_y = ud_mat.at<float>(i,1);
        float meter_depth  = meter_depth_mat.at<float>(d_y, d_x);
        if(meter_depth ==0)
        {
            continue;
        }
        Mat pixelPoint(3,1,CV_32F);
        pixelPoint.at<float>(0,0) = ud_mat.at<float>(i,0);
        pixelPoint.at<float>(1,0) = ud_mat.at<float>(i,1);
        pixelPoint.at<float>(2,0) = 1;
        Mat Point_with_depth = (inv_inst*pixelPoint)*meter_depth;
        Point3f objectPoint;
        objectPoint.x = Point_with_depth.at<float>(0,0);
        objectPoint.y = Point_with_depth.at<float>(1,0);
        objectPoint.z = Point_with_depth.at<float>(2,0);
        objectPoints.push_back(objectPoint);
        imagePixels.push_back(mp_second[i]);
    }
    Mat rvec;
    // cout<<"3차원 점의 수 :"<<objectPoints.size()<<endl;
    if(objectPoints.size()<20)
    {
        return false;
    }
    solvePnPRansac(objectPoints, imagePixels, _inst_param, _DistCoefParam, rvec, T);
    Rodrigues(rvec, R);//R+tvec 더하고, 투영되는 점 구하고, depth구함. 그리고 나서 depth 매칭함
    T.convertTo(T,CV_32FC1);
    R.convertTo(R,CV_32FC1);
    chrono::system_clock::time_point EndTime = chrono::system_clock::now();
    chrono::microseconds micro = chrono::duration_cast<chrono::microseconds>(EndTime - StartTime);
    return true;
}

void GeoMaskMaker::GetFlow(Mat &flow)
{
    int image_width =_firstImage.cols;
    int image_height = _firstImage.rows;
    Mat prvs, next;
    cvtColor(_firstImage, prvs, COLOR_BGR2GRAY);
    cvtColor(_secondImage, next, COLOR_BGR2GRAY);
    calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
}
void GeoMaskMaker::GetNoGMMmask(Mat &mask)
{
    cout<<image_count<<"checking"<<endl;
    image_count+=1;
    if(!start_flag)
    {
        mask = Mat::ones(mimage_height,mimage_width, CV_8UC1);
        return;
    }
    Mat R;
    Mat T;
    
    bool success = GetRt(R,T);
    if(!success)
    {
        mask = Mat::ones(mimage_height,mimage_width, CV_8UC1);
        cout<<"small feature match."<<endl;
        return;
    }
    
    //왜곡되지 않은 flow2개를 비교한다.
    Mat flow(mimage_height,mimage_width, CV_32FC2);
    GetFlow(flow);
    Mat dist_image = Mat::zeros(mimage_height,mimage_width,CV_32FC1);
    float R_0_0 = R.at<float>(0,0);
    float R_0_1 = R.at<float>(0,1);
    float R_1_0 = R.at<float>(1,0);
    float R_1_1 = R.at<float>(1,1);
    float R_2_0 = R.at<float>(2,0);
    float R_2_1 = R.at<float>(2,1);
    chrono::system_clock::time_point StartTime = chrono::system_clock::now();
    Mat EdgeImage_first =GetEdge(_firstDepth);
    Mat EdgeImage_second =GetEdge(_secondDepth);
    Mat inv_inst_param = _inst_param.inv();

    Mat J_sf= Mat::zeros(3,6,CV_32FC1);
    Mat S_sf = Mat::eye(6,6,CV_32FC1);
    Mat homoCurPixel(3,1,CV_32FC1);
    Mat homoRefPixel(3,1,CV_32FC1);

    Mat histogram_use_mask = Mat::zeros(mimage_height,mimage_width,CV_8UC1);
    for(int y = 0; y < mimage_height; ++y)
    {
        for(int x = 0; x < mimage_width; ++x)
        {
            //현재 픽셀점을 얻습니다.
            float cur_x = x +flow.at<Vec2f>(y,x)[0];
            float cur_y = y +flow.at<Vec2f>(y,x)[1];
            if(cur_x<0|cur_y<0|cur_x>mimage_width-1|cur_y>mimage_height-1)
            {//이미지 밖을 벗어나는 점은 버립니다.
                continue;
            }
            Point2f undi_refp = GetUndistortedPixel(y,x);//왜곡되지 않은 이전 점을 얻습니다.
            Point2f undi_curp = GetUndistortedPixel((int)cur_y,(int)cur_x);//왜곡되지 않은 현재 점을 얻습니다.
            //왜곡보정이 된 점을 토대로 depth값을 얻어옵니다.
            float ref_depth = _firstDepth.at<float>((int)undi_refp.y, (int)undi_refp.x);
            float cur_depth = _secondDepth.at<float>((int)undi_curp.y, (int)undi_curp.x);
            if(EdgeImage_first.at<uchar>((int)undi_refp.y, (int)undi_refp.x)==255|
            EdgeImage_second.at<uchar>((int)undi_curp.y, (int)undi_curp.x)==255)
            {
                continue;
            }
            if(cur_depth==0|cur_depth>3.5|ref_depth==0|ref_depth>3.5)
            {//깊이값이 부정확한 점은 버립니다. 
                continue;
            }
            
            homoCurPixel.at<float>(0,0) = undi_curp.x;
            homoCurPixel.at<float>(1,0) = undi_curp.y;
            homoCurPixel.at<float>(2,0) = 1;
            
            homoRefPixel.at<float>(0,0) = undi_refp.x;
            homoRefPixel.at<float>(1,0) = undi_refp.y;
            homoRefPixel.at<float>(2,0) = 1;
            Mat U_t = R*inv_inst_param*homoRefPixel;
            Mat CurPoint3D = inv_inst_param*homoCurPixel*cur_depth;
            Mat RefPoint3D = U_t*ref_depth+T;
            Mat dist_btw_Cur_Ref = CurPoint3D-RefPoint3D;
            
            S_sf.at<float>(2,2)= depth2std(ref_depth);
            S_sf.at<float>(5,5)= depth2std(cur_depth);
            
            
            J_sf.at<float>(0,0) = cur_depth/_fu;
            J_sf.at<float>(0,2) = (undi_curp.x-_cu)/_fu;
            J_sf.at<float>(0,3) = -R_0_0*ref_depth/_fu;
            J_sf.at<float>(0,4) = -R_0_1*ref_depth/_fv;
            J_sf.at<float>(0,5) = -U_t.at<float>(0,0);

            J_sf.at<float>(1,1) = ref_depth/_fv;
            J_sf.at<float>(1,2) = (undi_curp.x-_cu)/_fv;
            J_sf.at<float>(1,3) = -R_1_0*ref_depth/_fu;
            J_sf.at<float>(1,4) = -R_1_1*ref_depth/_fv;
            J_sf.at<float>(1,5) = -U_t.at<float>(1,0);

            J_sf.at<float>(2,2) = 1;
            J_sf.at<float>(2,3) = -R_2_0*ref_depth/_fu;
            J_sf.at<float>(2,4) = -R_2_1*ref_depth/_fv;
            J_sf.at<float>(2,5) = -U_t.at<float>(2,0);
            
            Mat likelihood = (dist_btw_Cur_Ref.t()*(J_sf*S_sf*J_sf.t()).inv()*dist_btw_Cur_Ref);
            float value = sqrt(likelihood.at<float>(0,0));//norm(dist_btw_Cur_Ref);
            dist_image.at<float>((int)cur_y,(int)cur_x) = value;
            histogram_use_mask.at<uchar>((int)cur_y,(int)cur_x) = 1;
        }
    }
    chrono::system_clock::time_point EndTime = chrono::system_clock::now();
    chrono::microseconds micro = chrono::duration_cast<chrono::microseconds>(EndTime - StartTime);
    
    normalize(dist_image,dist_image,0.,255.,NORM_MINMAX);
    dist_image.convertTo(dist_image,CV_8UC1);
    const int* channel_numbers = { 0 };
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = channel_range;
    int number_bins = 255;
    MatND histogram;
    calcHist(&dist_image, 1, channel_numbers, histogram_use_mask, histogram, 1, &number_bins, &channel_ranges);
    float hist_size = sum(histogram)[0];
    chrono::system_clock::time_point StartTime2 = chrono::system_clock::now();
    double mu = 0, scale = 1. / (hist_size);
    for (int i = 0; i < 255; i++)
        mu += i*(double)histogram.at<float>(i,0);

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;
    
    for (int i = 0; i < 255; i++)
    {
        double p_i, q2, mu2, sigma;

        p_i = histogram.at<float>(i,0) * scale;//각 확률
        mu1 *= q1;//
        q1 += p_i;//확률누적
        q2 = 1. - q1;//나머지 확률 

        if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON)
            continue;

        mu1 = (mu1 + i*p_i) / q1;
        mu2 = (mu - q1*mu1) / q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if (sigma > max_sigma)
        {
            max_sigma = sigma;
            max_val = i;
        }
    }
    chrono::system_clock::time_point EndTime2 = chrono::system_clock::now();
    chrono::microseconds micro2 = chrono::duration_cast<chrono::microseconds>(EndTime2 - StartTime2);
    int hist_w = dist_image.cols;//원본이미지의 사이즈를 얻어옵니다. 
    int hist_h =dist_image.rows;//원본이미지의 사이즈를 얻어옵니다. 
    int bin_w = cvRound((double)hist_w / number_bins);//각 bin의 사이즈

    Mat hist_img = Mat::zeros(hist_h, hist_w, CV_8UC3);
    
    normalize(histogram, histogram, 0, hist_img.rows, NORM_MINMAX, -1, Mat());
    
    
    int thresValue = 20;//max_val;
    //The original method uses actual distance thresholds rather than normalized thresholds.
    //
    // cv::Ptr<cv::ml::EM> gmm = cv::ml::EM::create();
    // gmm->set


    // Mat return_img = dist_image<thresValue;
    // cout<<return_img<<endl;
    // cvtColor(return_img,return_img,COLOR_GRAY2BGR);
    // for (int i = 1; i < number_bins; i++)
    // {
    //     Scalar main_color;
    //     if(i-1==thresValue)
    //     {
    //         main_color = Scalar(0, 0, 255);
    //         line(hist_img, Point(bin_w * (i - 1), 
    //                         0), 
    //                     Point(bin_w * (i-1), 
    //                         hist_h ), 
    //                     main_color, 
    //                     1, 8, 0);
    //     }
    //     else
    //     {
    //         main_color = Scalar(0, 255,0);
    //     }
    //     line(hist_img, Point(bin_w * (i - 1), 
    //                         hist_h - cvRound(histogram.at<float>(i - 1))), 
    //                     Point(bin_w * (i-1), 
    //                         hist_h ), 
    //                     main_color, 
    //                     1, 8, 0);
    // }
    // hconcat(hist_img,return_img,return_img);
    // imshow("HistImage_slider", return_img);
    // waitKey(1);


    //-------------------------------------------------------
    // namedWindow("HistImage_slider",1);
    // createTrackbar("threshold","HistImage_slider",&thresValue,255);
    // while('s' != waitKey(10))
    // {
    //     hist_img = Mat(hist_h, hist_w, CV_8UC3, Scalar::all(255));
    //     Mat return_img = dist_image>thresValue;
    //     cvtColor(return_img,return_img,COLOR_GRAY2BGR);
    //     for (int i = 1; i < number_bins; i++)
    //     {
    //         Scalar main_color;
    //         if(i-1==thresValue)
    //         {
    //             main_color = Scalar(0, 0, 255);
    //             main_color = Scalar(0, 0, 255);
    //         line(hist_img, Point(bin_w * (i - 1), 
    //                         0), 
    //                     Point(bin_w * (i-1), 
    //                         hist_h ), 
    //                     main_color, 
    //                     1, 8, 0);
    //         }
    //         else
    //         {
    //             main_color = Scalar(0, 255,0);
    //         }
    //         line(hist_img, Point(bin_w * (i - 1), 
    //                             hist_h - cvRound(histogram.at<float>(i - 1))), 
    //                         Point(bin_w * (i-1), 
    //                             hist_h ), 
    //                         main_color, 
    //                         1, 8, 0);
    //     }
    //     hconcat(hist_img,return_img,return_img);
    //     Mat return_img2;
    //     hconcat(_firstImage,_secondImage,return_img2);
    //     vconcat(return_img2,return_img,return_img);
    //     imshow("HistImage_slider", return_img);
    // }

    Mat return_img = dist_image<thresValue;
    return_img /=255;
    mask =return_img;
}
void GeoMaskMaker::AddNewImage(Mat new_Image,Mat new_Depth, Mat label, Mat originlabel)
{
    //새로 들어온 이미지를 vector에 저장합니다.
    Image_set newSet;
    new_Image.copyTo(newSet.RGB_image);
    // new_Depth.convertTo(new_Depth,CV_32FC1);
    // Mat result;
    new_Depth.copyTo(newSet.Depth_image);
    saved_Image_set.push_back(newSet);

    if(saved_Image_set.size()>inter_frame_size)//즉 사이즈가 6쯤 되면, 최종적으로 하나 줄인다.
    {
        // cout<<saved_Image_set.size()<<endl;
        saved_Image_set[0].RGB_image.copyTo(_firstImage);
        saved_Image_set[0].Depth_image.copyTo(_firstDepth);
        saved_Image_set[inter_frame_size].RGB_image.copyTo(_secondImage);
        saved_Image_set[inter_frame_size].Depth_image.copyTo(_secondDepth);
        saved_Image_set.erase(saved_Image_set.begin());
        start_flag = true;
    }
}
void GeoMaskMaker::GetGeoMask()
{
    if(!start_flag)
    {
        return;
    }
    imshow("first",_firstImage);float _cu;
    imshow("second",_secondImage);
    chrono::system_clock::time_point StartTime = chrono::system_clock::now();
    double scalefactor = 1.2;
    double pyramid_size = 8;
    const static auto& _orb_OrbHandle = ORB::create(2000,scalefactor,pyramid_size,31,0,2);

    Mat des_first;
    vector<KeyPoint>  kp_first;
    _orb_OrbHandle->detectAndCompute(_firstImage,noArray(),kp_first,des_first);

    Mat des_second;
    vector<KeyPoint>  kp_second;
    _orb_OrbHandle->detectAndCompute(_secondImage,noArray(),kp_second,des_second);

    vector<DMatch> matches;
    Ptr<DescriptorMatcher> _match_OrbMatchHandle = BFMatcher::create(NORM_HAMMING,true);
    _match_OrbMatchHandle->match(des_first,des_second,matches);
    sort(matches.begin(),matches.end());
    // cout<<"매칭 수"<<matches.size()<<endl;
    vector<DMatch> good_matchs(matches.begin(), matches.begin()+100);
    vector<Point2f> mp_first;
    vector<Point2f> mp_second;
    for(int i=0; i<good_matchs.size(); i++)
    {
        mp_first.push_back(kp_first[good_matchs[i].queryIdx].pt);
        mp_second.push_back(kp_second[good_matchs[i].trainIdx].pt);
    }
    Mat EssentialMat = findEssentialMat(mp_first, mp_second, _inst_param,RANSAC);
    // cout<<EssentialMat<<endl;
    vector<Point3f> mp_homo_first;
    vector<Point3f> mp_homo_second;
    for(int i=0; i<good_matchs.size(); i++)
    {
        mp_first.push_back(kp_first[good_matchs[i].queryIdx].pt);
        mp_second.push_back(kp_second[good_matchs[i].trainIdx].pt);
    }
    Mat ud_mat(mp_first.size(),2,CV_32F);
    for(int i=0; i<mp_first.size(); i++)
    {
        ud_mat.at<float>(i,0)=mp_first[i].x;
        ud_mat.at<float>(i,1)=mp_first[i].y;
    }
    ud_mat=ud_mat.reshape(2);
    undistortPoints(ud_mat,ud_mat,_inst_param,_DistCoefParam,cv::Mat(),_inst_param);
    //처음 꺼에 대해서, 왜곡 풀고, 깊이값 확인해서 있는점들만 매칭하면됨
    ud_mat=ud_mat.reshape(1);
    // cout<<ud_mat.rows<<" : "<<ud_mat.cols<<endl;
    vector<Point3f> objectPoints;
    vector<Point2f> imagePixels;

    //depth값 확인하면됨
    Mat meter_depth_mat = _firstDepth.clone();
    Mat inv_inst = _inst_param.inv();
    for(int i=0; i<mp_first.size(); i++)
    {
        int d_x = ud_mat.at<float>(i,0);
        int d_y = ud_mat.at<float>(i,1);
        float meter_depth  = meter_depth_mat.at<float>(d_y, d_x);
        if(meter_depth ==0|meter_depth>3.5)
        {
            continue;
        }
        Mat pixelPoint(3,1,CV_32F);
        pixelPoint.at<float>(0,0) = ud_mat.at<float>(i,0);
        pixelPoint.at<float>(1,0) = ud_mat.at<float>(i,1);
        pixelPoint.at<float>(2,0) = 1;
        Mat Point_with_depth = (inv_inst*pixelPoint)*meter_depth;
        Point3f objectPoint;
        objectPoint.x = Point_with_depth.at<float>(0,0);
        objectPoint.y = Point_with_depth.at<float>(1,0);
        objectPoint.z = Point_with_depth.at<float>(2,0);
        objectPoints.push_back(objectPoint);
        imagePixels.push_back(mp_second[i]);
    }
    Mat rvec, tvec;
    solvePnPRansac(objectPoints, imagePixels, _inst_param, _DistCoefParam, rvec, tvec);
    Mat R;
    Rodrigues(rvec, R);//R+tvec 더하고, 투영되는 점 구하고, depth구함. 그리고 나서 depth 매칭함.
    Mat transfrom_mat;

    hconcat(R,tvec,transfrom_mat);
    vector<float> depth_current3dPoints;//다음프레임에서 본 3차원점
    int image_width =_firstImage.cols;
    int image_height = _firstImage.rows;
    
    //이제 초기이미지의 3차원 점들을 depth 기반으로 재투영해보자.
    Mat ud_mat_2(image_height*image_width,2,CV_32F);
    int idx=0;
    for(int x = 0; x < image_width; ++x)
    {
        for(int y = 0; y < image_height; ++y)
        {
            ud_mat_2.at<float>(idx,0)=x;
            ud_mat_2.at<float>(idx,1)=y;
            idx++;
        }
    }
    Mat origin_point = ud_mat_2.clone();
    ud_mat_2=ud_mat_2.reshape(2);
    undistortPoints(ud_mat_2,ud_mat_2,_inst_param,_DistCoefParam,cv::Mat(),_inst_param);
    ud_mat_2=ud_mat_2.reshape(1);//이제 모든 이미지 픽셀에서 투영전에 왜곡이 보정된 픽셀좌표를 얻게됨.
    Mat result_image = Mat::zeros(image_height,image_width,CV_8UC3);//before projected
    vector<Point3f> good_depth_points;
    vector<Point3i> good_depth_colors;
    vector<Point2f> good_depth_first_pixel;
    Mat EdgeImage_first =GetEdge(_firstDepth);
    Mat EdgeImage_second =GetEdge(_secondDepth);
    for(int i=0; i<ud_mat_2.rows; i++)
    {
        int d_x = ud_mat_2.at<float>(i,0);
        int d_y = ud_mat_2.at<float>(i,1);
        if(EdgeImage_first.at<uchar>(d_y,d_x)==255)
        {
            continue;//원본이 물체의 경계에 해당하는영역이라면
        }
        float meter_depth  = meter_depth_mat.at<float>(d_y, d_x);
        int o_x = origin_point.at<float>(i,0);
        int o_y = origin_point.at<float>(i,1);
        if(meter_depth ==0|meter_depth>3.5)
        {
            continue;
        }
        //색복사함
        result_image.at<Vec3b>(d_y,d_x)[0] = _firstImage.at<Vec3b>(o_y,o_x)[0];
        result_image.at<Vec3b>(d_y,d_x)[1] = _firstImage.at<Vec3b>(o_y,o_x)[1];
        result_image.at<Vec3b>(d_y,d_x)[2] = _firstImage.at<Vec3b>(o_y,o_x)[2];

        Mat pixelPoint(3,1,CV_32F);
        pixelPoint.at<float>(0,0) = ud_mat_2.at<float>(i,0);
        pixelPoint.at<float>(1,0) = ud_mat_2.at<float>(i,1);
        pixelPoint.at<float>(2,0) = 1;
        Mat Point_with_depth = (inv_inst*pixelPoint)*meter_depth;
        Point3f objectPoint;
        objectPoint.x = Point_with_depth.at<float>(0,0);
        objectPoint.y = Point_with_depth.at<float>(1,0);
        objectPoint.z = Point_with_depth.at<float>(2,0);
        good_depth_points.push_back(objectPoint);
        good_depth_colors.push_back(Point3i(result_image.at<Vec3b>(d_y,d_x)[0],result_image.at<Vec3b>(d_y,d_x)[1],result_image.at<Vec3b>(d_y,d_x)[2]));
        good_depth_first_pixel.push_back(Point2f(o_x,o_y));
    }
    Mat testImage = Mat::zeros(image_height,image_width,CV_32FC1);
    Mat projectedDepthImage = Mat::zeros(image_height,image_width,CV_32FC1);
    transfrom_mat.convertTo(transfrom_mat,CV_32F);
    vector<Point2f> bad_depth_point2f;
    Mat origin_diff = Mat::zeros(image_height,image_width,CV_32FC1);
    for(int i =0; i<good_depth_points.size(); i++)
    {
        Mat objectPoint_ref = Mat(4,1,CV_32FC1);
        objectPoint_ref.at<float>(0,0) = good_depth_points[i].x;
        objectPoint_ref.at<float>(1,0) = good_depth_points[i].y;
        objectPoint_ref.at<float>(2,0) = good_depth_points[i].z;
        objectPoint_ref.at<float>(3,0) = 1;
        Mat objectPoint_cur = transfrom_mat*objectPoint_ref;//현재 프레임에서의 3차원점.
        Mat undistorted_pixel_point = _inst_param*objectPoint_cur;
        float x = undistorted_pixel_point.at<float>(0,0);
        float y = undistorted_pixel_point.at<float>(1,0);
        float z = undistorted_pixel_point.at<float>(2,0);
        x /=z;
        y /=z;
        if(x<0|y<0|x>image_width-1|y>image_height-1)
        {
            continue;
        }
        float real_depth =_secondDepth.at<float>((int)y,(int)x);
        if(real_depth ==0)
        {
            continue;
        }
        if(EdgeImage_second.at<uchar>((int)y,(int)x)==255)
        {
            continue;
        }
        
        float diff = real_depth - objectPoint_cur.at<float>(2,0);//예상되는 물체와 실제 물체사이의 거리
        if(diff<0)
        {
            diff = -diff;
            projectedDepthImage.at<float>(y,x) = objectPoint_cur.at<float>(2,0);
            testImage.at<float>(y,x) = diff;//depth차이만 구해온거
            origin_diff.at<float>(((int)good_depth_first_pixel[i].y),((int)good_depth_first_pixel[i].x))  = diff;
        }
        
    }
    double minVal; 
    double maxVal; 
    Point minLoc; 
    Point maxLoc;
    
    imshow("secondDepth",EdgeImage_second);
    minMaxLoc( testImage, &minVal, &maxVal, &minLoc, &maxLoc );

    cout << "min val: " << minVal << endl;
    cout << "max val: " << maxVal << endl;
    normalize(testImage,testImage,0,255,NORM_MINMAX);
    
    testImage.convertTo(testImage,CV_8UC1);
    normalize(origin_diff,origin_diff,0,255,NORM_MINMAX);
    origin_diff.convertTo(origin_diff,CV_8UC1);
    Mat bigDepth_change_image = testImage>10;
    Mat bigMove_image = origin_diff>30;
    Mat bad_point_Image = Mat::zeros(image_height,image_width,CV_8UC3);

    for(int i =0; i<image_height; i++)
    {
        for(int j=0; j<image_width; j++)
        {
            if(bigMove_image.at<uchar>(i,j) ==255)
            {
                bad_point_Image.at<Vec3b>(i,j)[0] = _secondImage.at<Vec3b>(i,j)[0];
                bad_point_Image.at<Vec3b>(i,j)[1] = _secondImage.at<Vec3b>(i,j)[1];
                bad_point_Image.at<Vec3b>(i,j)[2] = _secondImage.at<Vec3b>(i,j)[2];
            }
        }
    }

    imshow("bad_point_Image",bad_point_Image);
    imshow("depth_test",testImage);
    imshow("origin_diff",bigMove_image);
    vector<Point2f> pixel_projected_point;
    projectPoints(good_depth_points,rvec,tvec,_inst_param,_DistCoefParam,pixel_projected_point);//현재 프레임에 실제 투영된 좌표.이게 실제 depth를 의미하지는 않음...
    
    Mat projected_image = Mat::zeros(image_height,image_width,CV_8UC3);
    Mat prvs, next;
    cvtColor(_firstImage, prvs, COLOR_BGR2GRAY);
    cvtColor(_secondImage, next, COLOR_BGR2GRAY);
    
    Mat flow(prvs.size(), CV_32FC2);
    Mat largedepthdiffImage = Mat::zeros(image_height,image_width,CV_8UC3);
    calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    //Ptr<cv::DualTVL1OpticalFlow> duflow = createOptFlow_DualTVL1();
    //duflow->calc(prvs, next,flow);
    
    Mat dist_image = Mat::zeros(image_height,image_width,CV_32FC1);
    Mat optical_image = Mat::zeros(image_height,image_width,CV_8UC3);
    Mat projected_error_image = Mat::zeros(image_height,image_width,CV_32FC1);
    Mat mask_pixel = Mat::zeros(image_height,image_width,CV_8UC1);
    Mat optical_bigmove_image = Mat::zeros(image_height,image_width,CV_8UC3);
    for(int i=0; i<pixel_projected_point.size();i++)
    {
        int x = pixel_projected_point[i].x;
        int y = pixel_projected_point[i].y;
        if(x<0|y<0|x>image_width-1|y>image_height-1)
        {
            continue;
            
        }
        projected_image.at<Vec3b>(y,x)[0] = good_depth_colors[i].x;
        projected_image.at<Vec3b>(y,x)[1] = good_depth_colors[i].y;
        projected_image.at<Vec3b>(y,x)[2] = good_depth_colors[i].z;
        float color_error = (projected_image.at<Vec3b>(y,x)[0] - _secondImage.at<Vec3b>(y,x)[0])*
                            (projected_image.at<Vec3b>(y,x)[0] - _secondImage.at<Vec3b>(y,x)[0])+
                            (projected_image.at<Vec3b>(y,x)[1] - _secondImage.at<Vec3b>(y,x)[1])*
                            (projected_image.at<Vec3b>(y,x)[1] - _secondImage.at<Vec3b>(y,x)[1])+
                            (projected_image.at<Vec3b>(y,x)[2] - _secondImage.at<Vec3b>(y,x)[2])*
                            (projected_image.at<Vec3b>(y,x)[2] - _secondImage.at<Vec3b>(y,x)[2]);
        projected_error_image.at<float>(y,x) = color_error;//투영된 이미지사이의 빛의 차이


        int origin_flow_x = good_depth_first_pixel[i].x;
        int origin_flow_y = good_depth_first_pixel[i].y;
        float flow_x = origin_flow_x +flow.at<Vec2f>(origin_flow_y,origin_flow_x)[0];
        float flow_y = origin_flow_y +flow.at<Vec2f>(origin_flow_y,origin_flow_x)[1];
        float projected_depth = projectedDepthImage.at<float>(origin_flow_y,origin_flow_x);
        float flow_depth = _secondDepth.at<float>(flow_y,flow_x);
        Point3f real_point = Point3f(flow_x,flow_y,flow_depth);
        Point3f pred_point = Point3f(origin_flow_x,origin_flow_y,projected_depth);
        
        Point3f dist = (real_point-pred_point);
        //예상되는 깊이와 실제 관측값의 차이도 구해봐야하고, 실제 dense flow는 이미지의 경계에서 그런 해석이 어려우니 수정해보고
        double result = sqrt(dist.x*dist.x + dist.y*dist.y+ 100*dist.z*dist.z);
        //투영이 잘되었다면 이제는 뭐해야하지? 흠.. denseflow 비교해야하지않나?
        dist_image.at<float>(origin_flow_y,origin_flow_x) = result;
        optical_image.at<Vec3b>(flow_y,flow_x)[0] = _firstImage.at<Vec3b>(origin_flow_y,origin_flow_x)[0];
        optical_image.at<Vec3b>(flow_y,flow_x)[1] = _firstImage.at<Vec3b>(origin_flow_y,origin_flow_x)[1];
        optical_image.at<Vec3b>(flow_y,flow_x)[2] = _firstImage.at<Vec3b>(origin_flow_y,origin_flow_x)[2];
        // cout<<result<<endl;
        mask_pixel.at<uchar>(origin_flow_y,origin_flow_x) = 1;
        if(bigMove_image.at<uchar>(origin_flow_y,origin_flow_x) ==255)
        {
            optical_bigmove_image.at<Vec3b>(flow_y,flow_x)[0] = _firstImage.at<Vec3b>(origin_flow_y,origin_flow_x)[0];
            optical_bigmove_image.at<Vec3b>(flow_y,flow_x)[1] = _firstImage.at<Vec3b>(origin_flow_y,origin_flow_x)[1];
            optical_bigmove_image.at<Vec3b>(flow_y,flow_x)[2] = _firstImage.at<Vec3b>(origin_flow_y,origin_flow_x)[2];
        }
    }
    imshow("optical_bigmove_image",optical_bigmove_image);
    imshow("dist image2",dist_image);
    normalize(projected_error_image,projected_error_image,0.,255.,NORM_MINMAX);
    // cout<<dist_image<<endl;
    normalize(dist_image,dist_image,0.,255.,NORM_MINMAX);
    dist_image.convertTo(dist_image,CV_8UC1);
    const int* channel_numbers = { 0 };
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = channel_range;
    int number_bins = 255;
    MatND histogram;
    calcHist(&dist_image, 1, channel_numbers, mask_pixel, histogram, 1, &number_bins, &channel_ranges);
    int hist_w = dist_image.cols;
    int hist_h =dist_image.rows;
    int bin_w = cvRound((double)hist_w / number_bins);

    Mat hist_img(hist_h, hist_w, CV_8UC1, Scalar::all(0));
    Scalar hist_sum = sum(hist_img);
    
    normalize(histogram, histogram, 0, hist_img.rows, NORM_MINMAX, -1, Mat());//normalized된 값임 흠..

    for (int i = 1; i < number_bins; i++)
    {
        line(hist_img, Point(bin_w * (i - 1), 
                            hist_h - cvRound(histogram.at<float>(i - 1))), 
                        Point(bin_w * (i-1), 
                            hist_h ), 
                        Scalar(255, 0, 0), 
                        1, 8, 0);
    }
    // imshow("Histogram", hist_img);
    chrono::system_clock::time_point EndTime = chrono::system_clock::now();
    chrono::microseconds micro = chrono::duration_cast<chrono::microseconds>(EndTime - StartTime);
    // projectPoints()
    // cout<<"걸린시간 : "<<(micro.count())/1000000.<<"초"<<endl;
    // cout<<"결국 매칭길이 : "<<imagePixels.size()<<endl;
    Mat blended_image;
    addWeighted(projected_image,0.5,_secondImage,0.5,0,blended_image);
    // imshow("blending image",blended_image);
    imshow("projected_image",projected_image);
    imshow("dist image",dist_image);
    imshow("flow image", optical_image);
    // imshow("result image_2",projected_image);
    imshow("projected_error_image",projected_error_image);
    // imshow("reference frame",_firstImage);
    waitKey(0);
    
    
    // Undistort points
    // ud_mat=ud_mat.reshape(2);
    
    //_DistCoefParam, _inst_param 
}
void GeoMaskMaker::MakeGraph()
{
    if(_count==1)
    {
        return;
    }
    //1.Get segmented image by depth, Mask R-CNN result
    // 1-1 segmentation depth ?.. using kinect fusion? using opencv? 왜곡이 없는 마스킹 결과랑 depth결과를 이용하겟지
    // 1-2 how? no mind..
    //2.sample point from each segment
    // 2-1 uniform distribution( save pixel point in vector -> sample data(not edge, propotional to area) )
    //3.make graph
    // 3-1 using subdiv2d
    //4.get segmentation weight
    // 4-1 using segmentation information
    //5.get geometry weight ~~ can change
    // 5-1 using range sigma and normalization
    //6.cut graph using weight (see result)
    // 6-1 if segmentation is good, the segmetation with bad geometry have good result
    Mat FirstSeg;
    Mat SecondSeg;
    {
    chrono::system_clock::time_point StartTime = chrono::system_clock::now();
    FirstSeg = GetSegmentation(_firstRCNNMask,_firstDepth,_firstnum_label);
    chrono::system_clock::time_point EndTime = chrono::system_clock::now();
    chrono::microseconds micro = chrono::duration_cast<chrono::microseconds>(EndTime - StartTime);
    imshow("1seg",FirstSeg);
    FirstSeg.copyTo(_seg);
    cout<<"1.걸린시간 : "<<(micro.count())/1000000.<<"초"<<endl;
    }


    {
    chrono::system_clock::time_point StartTime = chrono::system_clock::now();
    SecondSeg = GetSegmentation(_secondRCNNMask,_secondDepth,_secondnum_label);
    chrono::system_clock::time_point EndTime = chrono::system_clock::now();
    chrono::microseconds micro = chrono::duration_cast<chrono::microseconds>(EndTime - StartTime);
    cout<<"2. 걸린시간 : "<<(micro.count())/1000000.<<"초"<<endl;
    imshow("2seg",SecondSeg);
    }


    vector<Point2i> samplePoint;
    {
    chrono::system_clock::time_point StartTime = chrono::system_clock::now();
    samplePoint = GetSamplePoint(FirstSeg,SecondSeg,_firstImage,_secondImage);
    chrono::system_clock::time_point EndTime = chrono::system_clock::now();
    chrono::microseconds micro = chrono::duration_cast<chrono::microseconds>(EndTime - StartTime);
    cout<<"3. 걸린시간 : "<<(micro.count())/1000000.<<"초"<<endl;
    }

    {
    chrono::system_clock::time_point StartTime = chrono::system_clock::now();
    GetBaseGraph(samplePoint);
    chrono::system_clock::time_point EndTime = chrono::system_clock::now();
    chrono::microseconds micro = chrono::duration_cast<chrono::microseconds>(EndTime - StartTime);
    cout<<"4. 걸린시간 : "<<(micro.count())/1000000.<<"초"<<endl;
    }
    waitKey(1);
    return;
}   

Mat GeoMaskMaker::GetMask()
{
    bool bMakeMask = false;
    if(_count>=2 & !bMakeMask)
    {
        return _secondRCNNMask.clone();
    }
    else if(_count>=2)
    {
        _Node.clear();
        _Edge.clear();
        return _return_mask.clone();
    }
    else
    {
        return _firstRCNNMask.clone();
    }
}
Mat GeoMaskMaker::GetEdge(Mat arg_Depth_image)
{
    int image_width = arg_Depth_image.cols;
    int image_height = arg_Depth_image.rows;
    Mat meter_depth = arg_Depth_image.clone();//*_DepthMapFactor;
    meter_depth.convertTo(meter_depth,CV_64FC1);
    Mat inst_param_d;
    _inst_param.convertTo(inst_param_d,CV_64FC1);

    //기하학적 먼저 세그멘테이션하자.
    Mat normals = Mat::zeros(meter_depth.size(), CV_64FC3);
    Mat vertex_points = Mat::zeros(meter_depth.size(), CV_64FC3);
    for(int x = 1; x < meter_depth.cols-1; ++x)
    {
        for(int y = 1; y < meter_depth.rows-1; ++y)
        {
            if(meter_depth.at<double>(y, x) >3.5)
            {
                meter_depth.at<double>(y, x) = 0.;
                continue;
            }
            if(meter_depth.at<double>(y-1, x) ==0.|meter_depth.at<double>(y, x) ==0.|meter_depth.at<double>(y, x-1) ==0.)
            {
                continue;
            }
            Vec3d t(x,y-1,meter_depth.at<double>(y-1, x)/*depth(y-1,x)*/);
            Vec3d l(x-1,y,meter_depth.at<double>(y, x-1)/*depth(y,x-1)*/);
            Vec3d c(x,y,meter_depth.at<double>(y, x)/*depth(y,x)*/);
            Vec3d d = (l-c).cross(t-c);
            Vec3d n = normalize(d);
            normals.at<Vec3d>(y,x) = n;
            //이제 3차원 좌표도 복원해와야함.
            Mat_<double> homo_pixel(3,1);
            homo_pixel << x,y,1;
            Mat homo_point3d = inst_param_d.inv()*homo_pixel;
            homo_point3d = homo_point3d*meter_depth.at<double>(y, x);
            Vec3d vec_point3d(homo_point3d.at<double>(0),homo_point3d.at<double>(1),homo_point3d.at<double>(2));
            vertex_points.at<Vec3d>(y,x) = vec_point3d;
        }
    }
    int neighbor_x[8] ={-1,-1,0,1,1,1,0,-1};
    int neighbor_y[8] ={0,-1,-1,-1,0,1,1,1};
    Mat edge_image = Mat::zeros(meter_depth.size(), CV_8UC1);
    for(int x = 1; x < meter_depth.cols-1; ++x)
    {
        for(int y = 1; y < meter_depth.rows-1; ++y)
        {
            if(meter_depth.at<double>(y, x) ==0.)
            {
                continue;
            }
            //이웃을 검사합니다.
            bool zero_neighbor_flag = false;
            double max_phi_d=-1;
            double max_phi_c =-1;
            for(int i=0; i<8; i++)
            {
                int p_nei_x = x+neighbor_x[i];
                int p_nei_y = y+neighbor_y[i];
                if(vertex_points.at<Vec3d>(p_nei_y,p_nei_x)[2] ==0)
                {
                    zero_neighbor_flag = true;
                }

                if(p_nei_x<0|p_nei_x>=image_width|p_nei_y<0|p_nei_y>=image_height|
                                    vertex_points.at<Vec3d>(p_nei_y,p_nei_x)[2] ==0)
                {
                    continue;
                }
                double phi_d = ((vertex_points.at<Vec3d>(p_nei_y,p_nei_x) - 
                                vertex_points.at<Vec3d>(y,x)).t()*normals.at<Vec3d>(y,x))[0];
                if(max_phi_d<abs(phi_d))
                {
                    max_phi_d = abs(phi_d);
                }
                double phi_c= 0;
                if(phi_d<0)
                {
                    if(max_phi_c<phi_c)
                    {
                        max_phi_c = phi_c;
                    }
                }
                else
                {
                    phi_c =1- (normals.at<Vec3d>(p_nei_y,p_nei_x).t()*normals.at<Vec3d>(y,x))[0];
                    if(phi_c>max_phi_c)
                    {
                        max_phi_c = phi_c;
                    }
                }
            }
            if(zero_neighbor_flag)
            {
                edge_image.at<uchar>(y,x) = 255;
                continue;
            }
            if(max_phi_c==-1|max_phi_d==-1)//주변에 depth가 0인구간이 있다면 포함합니다.
            {
                continue;
            }
            double thres_edge = max_phi_d+0.05*max_phi_c;
            //cout<<"thres_edge : "<<thres_edge<<endl;
            if(thres_edge>0.04)
            {
                edge_image.at<uchar>(y,x) = 255;
            }
        }
    }
    return edge_image;
}
Mat GeoMaskMaker::GetSegmentation(Mat arg_origin_label,Mat arg_Depth_image, int label_num)
{
    //모폴로지 연산을 해야하나? 
    int image_width = arg_origin_label.cols;
    int image_height = arg_origin_label.rows;
    Mat meter_depth = arg_Depth_image.clone();//*_DepthMapFactor;
    meter_depth.convertTo(meter_depth,CV_64FC1);
    Mat inst_param_d;
    _inst_param.convertTo(inst_param_d,CV_64FC1);

    //기하학적 먼저 세그멘테이션하자.
    Mat normals = Mat::zeros(meter_depth.size(), CV_64FC3);
    Mat vertex_points = Mat::zeros(meter_depth.size(), CV_64FC3);
    for(int x = 1; x < meter_depth.cols-1; ++x)
    {
        for(int y = 1; y < meter_depth.rows-1; ++y)
        {
            if(meter_depth.at<double>(y, x) >3.5)
            {
                meter_depth.at<double>(y, x) = 0.;
                continue;
            }
            if(meter_depth.at<double>(y-1, x) ==0.|meter_depth.at<double>(y, x) ==0.|meter_depth.at<double>(y, x-1) ==0.)
            {
                continue;
            }
            Vec3d t(x,y-1,meter_depth.at<double>(y-1, x)/*depth(y-1,x)*/);
            Vec3d l(x-1,y,meter_depth.at<double>(y, x-1)/*depth(y,x-1)*/);
            Vec3d c(x,y,meter_depth.at<double>(y, x)/*depth(y,x)*/);
            Vec3d d = (l-c).cross(t-c);
            Vec3d n = normalize(d);
            normals.at<Vec3d>(y,x) = n;
            //이제 3차원 좌표도 복원해와야함.
            Mat_<double> homo_pixel(3,1);
            homo_pixel << x,y,1;
            Mat homo_point3d = inst_param_d.inv()*homo_pixel;
            homo_point3d = homo_point3d*meter_depth.at<double>(y, x);
            Vec3d vec_point3d(homo_point3d.at<double>(0),homo_point3d.at<double>(1),homo_point3d.at<double>(2));
            vertex_points.at<Vec3d>(y,x) = vec_point3d;
        }
    }
    int neighbor_x[8] ={-1,-1,0,1,1,1,0,-1};
    int neighbor_y[8] ={0,-1,-1,-1,0,1,1,1};
    Mat edge_image = Mat::zeros(meter_depth.size(), CV_8UC1);
    for(int x = 1; x < meter_depth.cols-1; ++x)
    {
        for(int y = 1; y < meter_depth.rows-1; ++y)
        {
            if(meter_depth.at<double>(y, x) ==0.)
            {
                continue;
            }
            //이웃을 검사합니다.
            bool zero_neighbor_flag = false;
            double max_phi_d=-1;
            double max_phi_c =-1;
            for(int i=0; i<8; i++)
            {
                int p_nei_x = x+neighbor_x[i];
                int p_nei_y = y+neighbor_y[i];
                if(vertex_points.at<Vec3d>(p_nei_y,p_nei_x)[2] ==0)
                {
                    zero_neighbor_flag = true;
                }

                if(p_nei_x<0|p_nei_x>=image_width|p_nei_y<0|p_nei_y>=image_height|
                                    vertex_points.at<Vec3d>(p_nei_y,p_nei_x)[2] ==0)
                {
                    continue;
                }
                double phi_d = ((vertex_points.at<Vec3d>(p_nei_y,p_nei_x) - 
                                vertex_points.at<Vec3d>(y,x)).t()*normals.at<Vec3d>(y,x))[0];
                if(max_phi_d<abs(phi_d))
                {
                    max_phi_d = abs(phi_d);
                }
                double phi_c= 0;
                if(phi_d<0)
                {
                    if(max_phi_c<phi_c)
                    {
                        max_phi_c = phi_c;
                    }
                }
                else
                {
                    phi_c =1- (normals.at<Vec3d>(p_nei_y,p_nei_x).t()*normals.at<Vec3d>(y,x))[0];
                    if(phi_c>max_phi_c)
                    {
                        max_phi_c = phi_c;
                    }
                }
            }
            if(zero_neighbor_flag)
            {
                edge_image.at<uchar>(y,x) = 255;
                continue;
            }
            if(max_phi_c==-1|max_phi_d==-1)//주변에 depth가 0인구간이 있다면 포함합니다.
            {
                continue;
            }
            double thres_edge = max_phi_d+0.05*max_phi_c;
            //cout<<"thres_edge : "<<thres_edge<<endl;
            if(thres_edge>0.04)
            {
                edge_image.at<uchar>(y,x) = 255;
            }
        }
    }
    Mat inv_edge_image = 255-edge_image;
    Mat test_img = meter_depth !=0;//공간상에서 엣지이고, 0인부분들을은 유효하지 않으니, 빠지면 좋음 
    test_img = test_img & inv_edge_image;//공간상에서 엣지이고, 0인부분들을은 유효하지 않으니, 빠지면 좋음 
    // imshow("test img",test_img);
    return test_img;
}
vector<Point2i> GeoMaskMaker::GetSamplePoint(Mat sample_mask_first, Mat sample_mask_second, Mat first_image, Mat second_image)
{
    //initialize
    _Node.clear();
    _Edge.clear();


    //
    int image_width = sample_mask_first.cols;
    int image_height = sample_mask_first.rows;

    int sample_num =400;
    Mat seg_label;
    int numOfLables = connectedComponents(sample_mask_first, seg_label);
    seg_label.copyTo(_seglabel);
    cout<<"numOfLabels : "<<numOfLables<<endl;
    // cout<<"type : "<<type2str(seg_label.type())<<endl;
    // cout<<seg_label<<endl;
    vector<vector<Point2i>> candidate_points(numOfLables);
    for(int x = 0; x < image_width; ++x)
    {
        for(int y = 0; y < image_height; ++y)
        {
            if(sample_mask_first.at<uchar>(y,x) ==255)
            {
                candidate_points[seg_label.at<int>(y,x)-1].push_back(Point2i(x,y));
            }
        }
    }
    
    cout<<"크기 "<<candidate_points.size()<<endl;
    vector<Point2i> return_sample;
    for(int j=0; j<numOfLables; j++)
    {
        vector<Point2i> segment_sample = candidate_points[j];
        vector<Point2i> outputsample;
        int sample_size = segment_sample.size();//샘플 후보 사이즈 -> 전체중에서 1/5을 사용한다. 만약에 1/5이 10이하이면 그냥 다사용한다.
        int sample_num =0;
        if((int)(sample_size/5) <3)
        {//row sample
            //sample_num = sample_size;

        }
        else
        {
            sample_num = (int)(sample_size/10);
            if(sample_num>50)
            {
                sample_num = 50;
            }
            outputsample = GetSampleNum(segment_sample,sample_num);
            return_sample.insert(return_sample.end(),outputsample.begin(),outputsample.end());
        }
    }
    
    // sample(candidate_point.begin(),candidate_point.end(),back_inserter(outputsample),sample_num,std::mt19937{std::random_device{}()});
    Mat test_image = first_image.clone();
    
    Mat prvs, next;
    cvtColor(first_image, prvs, COLOR_BGR2GRAY);
    cvtColor(second_image, next, COLOR_BGR2GRAY);
    
    Mat flow(prvs.size(), CV_32FC2);
    calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    flow.copyTo(denseflow);
    vector<Point2i> return_sample_new;
    for(int i=0; i<return_sample.size(); i++)
    {
        Point2i cand_sample = return_sample[i];
        // cout<<"sample position"<<cand_sample<<endl;
        float x_move = flow.at<float>(cand_sample.y,cand_sample.x,0);
        float y_move = flow.at<float>(cand_sample.y,cand_sample.x,1);
        int x_next = cand_sample.x+x_move;
        int y_next = cand_sample.y+y_move;
        

        if(abs(x_move)>8|abs(y_move)>8|sample_mask_second.at<uchar>(y_next,x_next) ==0)
        {
            circle(test_image,return_sample[i],1,Scalar(0,255,0));
        }
        else
        {
            circle(test_image,return_sample[i],1,Scalar(0,0,255));
            return_sample_new.push_back(return_sample[i]);
        }
        
        // cout<<i<<" : "<<return_sample.size()<<endl;
    }
    imshow("circle",test_image);

    return return_sample_new;
}
vector<Point2i> GeoMaskMaker::GetSampleNum(vector<Point2i> full_sample,int s_num)
{
    vector<Point2i> return_sample;
    int max_ind = full_sample.capacity();
    for(int i=0; i<s_num; i++)
    {
        int random_value = rand()%max_ind;
        return_sample.push_back(full_sample[random_value]);
        full_sample.erase(full_sample.begin() + random_value);
        max_ind -=1;
    }
    return return_sample;
}
void GeoMaskMaker::GetBaseGraph(vector<Point2i> SamplePoint)
{
    p_divObjet =new Subdiv2D(Rect(0,0,_secondImage.cols,_secondImage.rows));//
    for(int i=0; i<SamplePoint.size(); i++)
    {
        int x1 = SamplePoint[i].x;
        int y1 = SamplePoint[i].y;
        p_divObjet->insert(((Point2f)SamplePoint[i]));
    }
    vector<Vec4f> edge_line;
    p_divObjet->getEdgeList(edge_line);
    Mat vis_image;
    cvtColor(_seg,vis_image,COLOR_GRAY2BGR);
    Mat line_image = vis_image.clone();
    Mat line_image_2 = vis_image.clone();
    Mat line_image_origin = vis_image.clone();
    Mat line_image_inner = vis_image.clone();
    Mat line_image_inner_2 = vis_image.clone();

    Mat node_check_map = Mat::zeros(_secondImage.size(),CV_64FC1);
    int Node_idx=1;
    int Edge_idx=0;
    Mat prvs, next;
    cvtColor(_firstImage, prvs, COLOR_BGR2GRAY);
    cvtColor(_secondImage, next, COLOR_BGR2GRAY);
    
    Mat flow(prvs.size(), CV_32FC2);
    calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    
    for(int i=0; i<edge_line.size(); i++)
    {
        int x1 = edge_line[i](0);
        int y1 = edge_line[i](1);
        int x2 = edge_line[i](2);
        int y2 = edge_line[i](3);
        if(x1 <=0|x1 >=_secondImage.cols-1|y1 <=0|y1 >=_secondImage.rows-1)
        {
            // cout<<"x1 "<<x1<<", y1 "<<y1<<endl;
            continue;
        }
        if(x2 <=0|x2 >=_secondImage.cols-1|y2 <=0|y2 >=_secondImage.rows-1)
        {
            // cout<<"x2 "<<x1<<", y2 "<<y1<<endl;
            continue;
        }
        if(node_check_map.at<double>(y1,x1)==0)
        {
            node_check_map.at<double>(y1,x1)=Node_idx;
            Node_GD newNode;
            newNode.nodeIdx = Node_idx;
            newNode.first_coord = Point2f(x1,y1);
            const Point2f floatatx1y1 = flow.at<Point2f>(y1,x1);
            newNode.second_coord = Point2f(x1+floatatx1y1.x,y1+floatatx1y1.y);
            _Node.push_back(newNode);
            Node_idx++;
            
        }
        if(node_check_map.at<double>(y2,x2)==0)
        {
            node_check_map.at<double>(y2,x2)=Node_idx;
            Node_GD newNode;
            newNode.nodeIdx = Node_idx;
            newNode.first_coord = Point2f(x2,y2);
            const Point2f floatatx2y2 = flow.at<Point2f>(y2,x2);
            newNode.second_coord = Point2f(x2+floatatx2y2.x,y2+floatatx2y2.y);
            _Node.push_back(newNode);
            Node_idx++;
        }
        Edge_GD newEdge;
        newEdge.edgeIdx =Edge_idx;
        newEdge.node_I = node_check_map.at<double>(y1,x1);
        newEdge.node_J = node_check_map.at<double>(y2,x2);
        _Edge.push_back(newEdge);
        _Node[newEdge.node_I-1].edgeIdxs.push_back(Edge_idx);
        _Node[newEdge.node_J-1].edgeIdxs.push_back(Edge_idx);
        Edge_idx +=1;
        line(line_image_origin,Point(x1,y1),Point(x2,y2),Scalar(0,0,255),1);
        // p_divObjet->initDelaunay
    }
    //distortion해서 나온 좌표에 대해서, depth를 사용한다. (1. undistortion -> 2. use depth)
    //1. 보정전에 denseflow 위치를 구한다.(v)
    //2. 처음과 나중프레임에서의 왜곡이 없어진 좌표를 구하고, 그에 대한 depth를 곱해서 3차원좌표를 복원한다.
    //3. 처음프레임에서의 거리값과 나중프레임에서의 거리값을 구한 뒤에 뺀다. 
    //4. 이제 각 좌표의 거리에 따른 sigma를 구하고, 더해서 분산을 구한다.
    //5. 정규화 한 값의 우도를 구한후에 그에 따른 밝기를 그래프상에서 나타낸다.
    int Node_size_x_2 = _Node.size()*2;
    
    Mat ud_mat(Node_size_x_2,2,CV_32F);
    for(int i=0; i<Node_size_x_2/2; i++)
    {
        ud_mat.at<float>(2*i,0)=_Node[i].first_coord.x;
        ud_mat.at<float>(2*i,1)=_Node[i].first_coord.y;
        ud_mat.at<float>(2*i+1,0)=_Node[i].second_coord.x;
        ud_mat.at<float>(2*i+1,1)=_Node[i].second_coord.y;
    }
    
    // Undistort points
    ud_mat=ud_mat.reshape(2);
    undistortPoints(ud_mat,ud_mat,_inst_param,_DistCoefParam,cv::Mat(),_inst_param);
    ud_mat=ud_mat.reshape(1);//(Node_size_x_2 2)
    Mat one_Add = Mat::ones(Node_size_x_2,1,CV_32F);
    Mat homo_mat;
    
    hconcat(ud_mat,one_Add,homo_mat);
    homo_mat = homo_mat.t();
    Mat pointsFace;
    pointsFace = (_inst_param.inv())*homo_mat;//depth만 곱하면, 3차원좌표입니다.
    // Mat 3d_Points = 
    // // Fill undistorted keypoint vector
    for(int i=0; i<Node_size_x_2/2; i++)
    {
        //왜곡보정이 된점이 화면밖일 수도 있다..혹은 depth가 없거나
        int f_x = ud_mat.at<float>(2*i,0);
        int f_y = ud_mat.at<float>(2*i,1);
        int s_x = ud_mat.at<float>(2*i+1,0);
        int s_y = ud_mat.at<float>(2*i+1,1);
        float f_depth = _firstDepth.at<float>(f_y,f_x);//distortion이 제거된 점에대한 픽셀로 얻는다.
        float s_depth = _secondDepth.at<float>(s_y,s_x);
        Mat point3df = pointsFace.col(2*i)*f_depth;//각각의 3차원점을 복원합니다.
        Mat point3ds = pointsFace.col(2*i+1)*s_depth;//각각의 3차원점을 복원합니다.
        _Node[i].first_depth = f_depth;
        _Node[i].second_depth = s_depth;
        _Node[i].first_3dPoint = Point3f(point3df.at<float>(0),point3df.at<float>(1),point3df.at<float>(2));
        _Node[i].second_3dPoint = Point3f(point3ds.at<float>(0),point3ds.at<float>(1),point3ds.at<float>(2));
    }

    int Edge_size = _Edge.size();
    for(int idx=0; idx<Edge_size; idx++)
    {
        //Edge로부터 노드정보를 가져온다.
        int node_i = _Edge[idx].node_I-1;
        int node_j = _Edge[idx].node_J-1;
        float obs_first = euclideanDist(_Node[node_i].first_3dPoint,_Node[node_j].first_3dPoint);
        float obs_second = euclideanDist(_Node[node_i].second_3dPoint,_Node[node_j].second_3dPoint);
        float result_obs_first = obs_first-obs_second;
        float s1 =depth2std(_Node[node_i].first_depth);
        float s2 =depth2std(_Node[node_i].second_depth);
        float s3 =depth2std(_Node[node_j].first_depth);
        float s4 =depth2std(_Node[node_j].second_depth);
        float sum_sigma = sqrt(s1*s1+s2*s2+s3*s3+s4*s4);
        // cout<<result_obs_first<<endl;
        //cout<<"sigma : "<<sum_sigma<<endl;
        float z = result_obs_first/sum_sigma;
        float likelihood = (1/(sqrt(M_PI*2)))*exp(-z*z/2);
        _Edge[idx].wegiht_g = likelihood/0.40;
        //cout<<"likelihood : "<<likelihood<<endl;
    }
    for(int i=0; i<Edge_size; i++)
    {
        int node_i = _Edge[i].node_I-1;
        int node_j = _Edge[i].node_J-1;
        int x1 = _Node[node_i].first_coord.x;
        int y1 = _Node[node_i].first_coord.y;
        int x2 = _Node[node_j].first_coord.x;
        int y2 = _Node[node_j].first_coord.y;
        float weight = _Edge[i].wegiht_g;
        if(weight>0.5)
        {
            if(_seglabel.at<int>(y1,x1) != _seglabel.at<int>(y2,x2))
            {
                Scalar color(0,0,255);//int(255*(1-weight))
                line(line_image,Point(x1,y1),Point(x2,y2),color,3);
            }
            else
            {
                Scalar color(0,250,0);//int(255*(1-weight))
                line(line_image_inner,Point(x1,y1),Point(x2,y2),color,3);
            }
        }
        else
        {
            if(_seglabel.at<int>(y1,x1) != _seglabel.at<int>(y2,x2))
            {
                Scalar color(255,0,0);//int(255*(1-weight))
                line(line_image_2,Point(x1,y1),Point(x2,y2),color,3);//서로 다른 원소라면
            }
            else
            {
                Scalar color(221,0,255);//int(255*(1-weight))
                line(line_image_inner_2,Point(x1,y1),Point(x2,y2),color,3);
            }
            
            
        }
        
    }
    imshow("inter good geometry",line_image);
    imshow("inter bad geometry",line_image_2);
    imshow("line image_origin",line_image_origin);
    imshow("inner good geometry",line_image_inner);
    imshow("inner bad geometry",line_image_inner_2);
    waitKey(0);
    // exit(0);
}

float euclideanDist(cv::Point3f& a, cv::Point3f& b)
{
    cv::Point3f diff = a - b;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
}
float GeoMaskMaker::depth2std(float depth)
{
    float sigma_norm = 0.5;
    
    return (1/_fu)*(1/_fu)*sigma_norm*sigma_norm*depth*depth*depth*depth;
}