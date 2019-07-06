# Research on Perception Module in Autonomous Driving

本人刚刚从事无人驾驶中感知模块的研究，目前主要研究采用多传感器融合进行3D目标检测，现将研究过程中研读的论文采用中英文对照翻译的形式整理在此仓库中。

## 3D Object Detection

自动驾驶汽车(Autonomous Vehicle, AV)需要准确感知其周围环境才能可靠运行，物体检测是这种感知系统的基本功能。自动驾驶中2D目标检测方法的性能已经得到很大的提高，在KITTI物体检测基准上实现了超过90%的平均精度(Average Precision, AP)。3D目标检测相比于2D目标检测仍然存在很大的性能差距，并且3D目标检测任务对于其它自动驾驶任务至关重要，如路径规划、碰撞避免，因此有必要对3D目标检测方法进行进一步的研究。表1-1显示了2D和3D目标检测的对比：

<div align=center><img width=70% height=70% src="/image/1-1.png" alt="2D和3D目标检测的对比"/></div>


自动驾驶车辆使用的传感器种类繁多，但大多数关于AV的研究都集中在相机和激光雷达上。相机分为被动式的单目相机和立体相机以及主动式的TOF相机。单目相机以像素强度的形式提供详细信息，其优点是以更大的尺度显示目标物体的形状和纹理特性，其缺点是缺少深度信息、且易受光照和天气条件影响；立体相机可用于恢复深度通道，但是匹配算法增加了计算处理负担；TOP相机提供的深度信息是通过测量调制的红外脉冲的飞行时间，优点是计算复杂度较低但缺点是分辨率较低。激光雷达传感器作为有源传感器通过发射激光束并测量从发射到接收的时间得到3D位置，其产生的是一组3D点组成的3D点云(Point Cloud),这些点云通常是稀疏且空间分布不均，优点是在恶劣天气和极端光照条件下仍能可靠运作。目前的价格问题仍是阻碍其大规模民用的主要原因之一。表1-2显示了AV中常用的几种传感器的对比。

<div align=center><img width=70% height=70% src="/image/1-2.png" alt="传感器对比"/></div>


3D目标检测方法根据输入的数据类型分为基于视觉图像方法，基于激光雷达点云方法，基于多模态信息融合方法。其中，基于视觉图像方法主要研究基于单目图像方法，基于激光雷达点云的方法又可分为三个之类，基于投影、体素和PointNet方法。这三种方法的详细对比见表1-3。

<div align=center><img width=90% height=90% src="/image/1-3.png" alt="3D目标检测方法对比"/></div>


### 1. 基于视觉图像方法

<div align=center><img width=80% height=80% src="/image/1-4.png" alt="基于单目视觉方法总结"/></div>

### 2. 基于激光雷达点云方法


### 3. 基于多模态信息融合方法

## Reference

* 综述文章: [1]
- 基于视觉图像方法: [2] [3] [4] [5] [6] [7] [8]
* 基于激光雷达点云方法: 
- 基于多模态信息融合方法: 

[1] Arnold E, Al-Jarrah O Y, Dianati M, et al. A survey on 3d object detection methods for autonomous driving applications[J]. IEEE Transactions on Intelligent Transportation Systems, 2019.

[2] Chen X, Kundu K, Zhu Y, et al. 3d object proposals for accurate object class detection[C]//Advances in Neural Information Processing Systems(NIPS). 2015: 424-432.

[3] Chen X, Kundu K, Zhang Z, et al. Monocular 3d object detection for autonomous driving[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition(CVPR). 2016: 2147-2156.

[4] Xiang Y, Choi W, Lin Y, et al. Data-driven 3d voxel patterns for object category recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition(CVPR). 2015: 1903-1911.

[5] Xiang Y, Choi W, Lin Y, et al. Subcategory-aware convolutional neural networks for object proposals and detection[C]//2017 IEEE winter conference on applications of computer vision (WACV). IEEE, 2017: 924-933.

[6] Chabot F, Chaouch M, Rabarisoa J, et al. Deep manta: A coarse-to-fine many-task network for joint 2d and 3d vehicle analysis from monocular image[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition(CVPR). 2017: 2040-2049.

[7] Mousavian A, Anguelov D, Flynn J, et al. 3d bounding box estimation using deep learning and geometry[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition(CVPR). 2017: 7074-7082.

[8] Payen de La Garanderie G, Atapour Abarghouei A, Breckon T P. Eliminating the Blind Spot: Adapting 3D Object Detection and Monocular Depth Estimation to 360° Panoramic Imagery[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 789-807.



