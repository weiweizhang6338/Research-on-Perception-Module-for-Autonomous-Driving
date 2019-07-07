# Research on Perception Module in Autonomous Driving

本人刚刚从事无人驾驶中感知模块的研究，目前主要研究采用多传感器融合进行3D目标检测，现将研究过程中研读的论文采用中英文对照翻译的形式整理在此仓库中。

## 3D Object Detection[1]

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

### Reference

* 综述文章: [1]
- 基于视觉图像方法: [2] [3] [4] [5] [6] [7] [8]
* 基于激光雷达点云方法: 
- 基于多模态信息融合方法: 

[1] Arnold E, Al-Jarrah O Y, Dianati M, et al. A survey on 3d object detection methods for autonomous driving applications[J]. IEEE Transactions on Intelligent Transportation Systems, 2019.

[2] Chen X, Kundu K, Zhu Y, et al. 3d object proposals for accurate object class detection[C]//Advances in Neural Information Processing Systems(NIPS). 2015: 424-432.

......Chen X , Kundu K , Zhu Y , et al. 3D Object Proposals using Stereo Imagery for Accurate Object Class Detection[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017:1-1.

[3] Chen X, Kundu K, Zhang Z, et al. Monocular 3d object detection for autonomous driving[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition(CVPR). 2016: 2147-2156.

[4] Xiang Y, Choi W, Lin Y, et al. Data-driven 3d voxel patterns for object category recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition(CVPR). 2015: 1903-1911.

[5] Xiang Y, Choi W, Lin Y, et al. Subcategory-aware convolutional neural networks for object proposals and detection[C]//2017 IEEE winter conference on applications of computer vision (WACV). IEEE, 2017: 924-933.

[6] Chabot F, Chaouch M, Rabarisoa J, et al. Deep manta: A coarse-to-fine many-task network for joint 2d and 3d vehicle analysis from monocular image[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition(CVPR). 2017: 2040-2049.

[7] Mousavian A, Anguelov D, Flynn J, et al. 3d bounding box estimation using deep learning and geometry[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition(CVPR). 2017: 7074-7082.

[8] Payen de La Garanderie G, Atapour Abarghouei A, Breckon T P. Eliminating the Blind Spot: Adapting 3D Object Detection and Monocular Depth Estimation to 360° Panoramic Imagery[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 789-807.


## CNN

卷积神经网络(Convolutional Neural Network, CNN)是深度学习领域重要的神经网络框架之一，也在计算机视觉领域中发挥着重要的作用。CNN从90年代的LeNet开始，21世纪初沉寂了10年，其锋芒被SVM等手工设计的特征盖过，直到12年AlexNet开始又再焕发第二春，从ZF Net到VGG，GoogLeNet再到ResNet和DenseNet，网络越来越深，架构越来越复杂，解决反向传播时梯度消失的方法也越来越巧妙。

### 1. LeNet5(1998)[1]——开山之作

LeNet 诞生于 1994 年，是最早的卷积神经网络之一，并且推动了深度学习领域的发展。自从 1988 年开始，在许多次成功的迭代后，这项由 Yann LeCun 完成的开拓性成果被命名为 LeNet5。LeNet5 的架构基于这样的观点：（尤其是）图像的特征分布在整张图像上，以及带有可学习参数的卷积是一种用少量参数在多个位置上提取相似特征的有效方式。在那时候，没有 GPU 帮助训练，甚至 CPU 的速度也很慢。因此，能够保存参数以及计算过程是一个关键进展。这和将每个像素用作一个大型多层神经网络的单独输入相反。LeNet5 阐述了那些像素不应该被使用在第一层，因为图像具有很强的空间相关性，而使用图像中独立的像素作为不同的输入特征则利用不到这些相关性。

<div align=center><img width=90% height=90% src="/image/2-1.png" alt="LeNet5网络架构"/></div>

LeNet5 一共有7层，不包括输入层(28x28):层C1是卷积层，层S2是池化层，层C3是卷积层，层S4是池化层，层C5是卷积层，层F6是全连接层，最后是输出层(10个类别)。LeNet的网络结构虽然简单，但是五脏俱全，卷积层、池化层、全连接层，这些都是现代CNN网络的基本组件。

发展瓶颈：LeNet5的设计较为简单，因此其处理复杂数据的能力有限；近来的研究发现，全连接层的计算代价过大，使用全卷积层组成的神经网络成为趋势。

### 2. AlexNet(2012)[2]——王者归来

2012年，Alex Krizhevsky发布了AlexNet，这是一个更深入，更广泛的LeNet版本，在ImageNet竞赛中以超过第二名10.9个百分点的绝对优势一举夺冠，从此深度学习和卷积神经网络名声鹊起，深度学习的研究如雨后春笋般出现，AlexNet的出现可谓是卷积神经网络的王者归来。

<div align=center><img width=90% height=90% src="/image/2-2.png" alt="AlexNet网络架构"/></div>

AlexNet输入尺寸为227x227x3，有5个卷积层，3个池化层，2个全连接层和1个输出层，输出1000个类别的概率。

AlexNet将LeNet扩展为更大的神经网络，可用于学习更复杂的对象和对象层次结构。这项工作的贡献是：
  * 成功使用ReLU作为CNN的激活函数，成功解决了Sigmoid在网络较深时的梯度弥散问题
  - 使用Dropout随机忽略一部分神经元，以避免模型过拟合
  * 全部选用最大池化，避免平均池化的模糊化效果
  - 使用GPU加速深度卷积网络的训练
