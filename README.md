# GD-SLAM
implements "On combining visual SLAM and dense scene flow to increase the robustness of localization and mapping in dynamic environments,"  with DynaSLAM framework

: DynaSLAM 프레임워크를 사용한 On combining visual SLAM and dense scene flow to increase the robustness of localization and mapping in dynamic environments
의 구현 코드입니다. 논문의 내용과 다르게 구현된 점은 stereo카메라가 아닌 Depth카메라를 이용하여 구현하였으며, 마할로노비스 거리를 구할 때 사용한 관측요소는 denseflow를 통한 픽셀좌표와
depth 6입니다. 기존의 카메라의 상대포즈에 대한 관측성분 6개는 제외하고, 공분산을 구하였습니다. 또한 고정된 거리의 임계값 대신에 현재프레임에 대한 마할로노비스 거리를 min-max 노멀라이즈하고, 그값에 임계값을 두었습니다.

## 사용법

### 빌드하기 
#### 준비하기
##### 라이브러리
c++ (DynaSLAM)과 동일합니다. 
- Install ORB-SLAM2 prerequisites: C++11 or C++0x Compiler, Pangolin, OpenCV and Eigen3  (https://github.com/raulmur/ORB_SLAM2).
- Install boost libraries with the command `sudo apt-get install libboost-all-dev`.

Python
- Install python 2.7, keras and tensorflow, and download the `mask_rcnn_coco.h5` model from this GitHub repository: https://github.com/matterport/Mask_RCNN/releases. 


#### 빌드시작하기
```
cd GD-SLAM
chmod +x build.sh
./build.sh
```
#### 사용하기 
: 기존 DynaSLAM과 동일합니다. RGB-D 예제만 사용가능합니다.
- Place the `mask_rcnn_coco.h5` model in the folder `GD-SLAM/src/python/`.
## RGB-D Example on TUM Dataset
- Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

- Associate RGB images and depth images executing the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools):

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```
These associations files are given in the folder `./Examples/RGB-D/associations/` for the TUM dynamic sequences.

- Execute the following command. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER` to the uncompressed sequence folder. Change `ASSOCIATIONS_FILE` to the path to the corresponding associations file. `PATH_TO_MASKS` and `PATH_TO_OUTPUT` are optional parameters.

  ```
  ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE (PATH_TO_MASKS) (PATH_TO_OUTPUT)
  ```
  
If `PATH_TO_MASKS` and `PATH_TO_OUTPUT` are **not** provided, only the geometrical approach is used to detect dynamic objects. 

If `PATH_TO_MASKS` is provided, Mask R-CNN is used to segment the potential dynamic content of every frame. These masks are saved in the provided folder `PATH_TO_MASKS`. If this argument is `no_save`, the masks are used but not saved. If it finds the Mask R-CNN computed dynamic masks in `PATH_TO_MASKS`, it uses them but does not compute them again.

If `PATH_TO_OUTPUT` is provided, the inpainted frames are computed and saved in `PATH_TO_OUTPUT`.


## reference
### DynaSLAM : PushyamiKaveti의 pushyami-dev branch 

사용이유 : 원래 DynaSLAM이 conversion.cc에서 python의 numpy object를 Mat으로 바꿀 때 에러가 났고, 해당 문제를 해결함.

해당 브랜치 : https://github.com/PushyamiKaveti/DynaSLAM/tree/pushyami-dev

Bescos, Berta & Facil, Jose & Civera, Javier & Neira, Jose. (2018). DynaSLAM: Tracking, Mapping and Inpainting in Dynamic Scenes. 3. 1-1. 10.1109/LRA.2018.2860039. 

### On combining visual SLAM and dense scene flow to increase the robustness of localization and mapping in dynamic environments

사용방식 : 기존논문의 stereo 카메라를 사용하여 disparity를 이용했던 것 대신에, depth카메라를 사용함

P. F. Alcantarilla, J. J. Yebes, J. Almazán and L. M. Bergasa, "On combining visual SLAM and dense scene flow to increase the robustness of localization and mapping in dynamic environments," 2012 IEEE International Conference on Robotics and Automation, 2012, pp. 1290-1297
