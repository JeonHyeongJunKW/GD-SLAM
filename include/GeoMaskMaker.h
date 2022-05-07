#ifndef GEOMASKMAKER_H_

#define GEOMASKMAKER_H_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>


using namespace cv;
using namespace std;
string type2str(int type);
float euclideanDist(cv::Point3f& a, cv::Point3f& b);
struct Node_GD
{
    Point2f first_coord;
    Point2f second_coord;
    Point3f first_3dPoint;
    Point3f second_3dPoint;
    vector<int> edgeIdxs;

    bool IsDynamicbySemantic = false;//의미론적으로 동적인 점인지 확인한다.
    int nodeIdx;
    float first_depth;
    float second_depth;
};
struct Edge_GD
{
    int edgeIdx;
    int node_I;
    int node_J;
    float weight_s;//semantic한 정보를 기반으로한 weight
    float wegiht_g;//기하학적 일관성을 기반으로한 weight
    float norm_weight;//주변데이터를 기반으로 얻은 weight
};
struct CalcSet
{
    Point2f distortedCurPixel;//왜곡후 현재 pixel
    Point2f distortedPrevPixel;//왜곡후 이전 pixel
    Point2f undistortedCurPixel;
    Point2f undistortedPrevPixel;
    float depth_cur;//현재 깊이
    float depth_prev;//이전 깊이

};
struct Image_set
{
    Mat RGB_image;
    Mat Depth_image;
};
class GeoMaskMaker
{
    public:
    int inter_frame_size = 5;
    Mat _inst_param;
    Mat _DistCoefParam;
    float _DepthMapFactor;
    vector<Vec3b> random_color;
    vector<Image_set> saved_Image_set;
    bool start_flag = false;

    vector<Node_GD> _Node;
    vector<Edge_GD> _Edge;
    Mat _firstImage;
    Mat _secondImage;
    
    Mat _firstDepth;
    Mat _secondDepth;
    Mat _firstRCNNMask;
    Mat _secondRCNNMask;
    Mat _return_mask;
    int _count = 0;
    int _inter_count =0;
    Subdiv2D* p_divObjet;
    Mat denseflow;
    float _fu;
    float _fv;
    float _cu;
    float _cv;

    public:
    int mimage_height;
    int mimage_width;
    Mat _firstOriginLabel;
    int _firstnum_label;
    Mat _secondOriginLabel;
    int _secondnum_label;

    Mat _seglabel;
    int image_count=0;
    
    public:
    GeoMaskMaker(Mat inst_param, Mat DistCoef, float DepthMapFactor);
    
    void AddNewImage(Mat new_RGB,Mat new_Depth, Mat label, Mat originlabel);
    Mat GetSegmentation(Mat arg_origin_label,Mat arg_Depth_image, int label_num);//세그멘테이션 결과가 어떻게 나오길 원하는거지? 엣지만 표시되고, 나머지는 연결요소로 구하면되는건가?
    void GetNoGMMmask(Mat &mask);//prototype2 . Mahalanobis
    bool GetRt(Mat &R, Mat &T);
    void GetFlow(Mat &flow);
    void GetGeoMask();//prototype1 . euclidan error
    Mat undistortedPoint;
    Point2f GetUndistortedPixel(int i, int j);

    Mat GetEdge(Mat arg_Depth_image);

    void MakeGraph();
    Mat GetMask();
    float depth2std(float depth);
    vector<Point2i> GetSamplePoint(Mat sample_mask_first, Mat sample_mask_second, Mat first_image, Mat second_image);
    vector<Point2i> GetSampleNum(vector<Point2i> full_sample,int s_num);
    void GetBaseGraph(vector<Point2i> SamplePoint);

    //for visualization
    Mat _seg;
};

/*
void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}
*/

#endif /* GEOMASKMAKER_H_ */