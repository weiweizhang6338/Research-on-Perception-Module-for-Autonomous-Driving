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

卷积神经网络(Convolutional Neural Network, CNN)是深度学习领域重要的神经网络框架之一，也在计算机视觉领域中发挥着重要的作用。CNN从90年代的LeNet开始，21世纪初沉寂了10年，其锋芒被SVM等手工设计的特征盖过，直到12年AlexNet开始又再焕发第二春，从ZF Net到VGG，GoogLeNet再到ResNet和DenseNet，网络越来越深，架构越来越复杂，解决反向传播时梯度消失的方法也越来越巧妙。下图显示了CNN的发展脉络。

<div align=center><img width=90% height=90% src="/image/2-0.png" alt="CNN发展脉络"/></div>

### 1. LeNet5(1998)[1]——开山之作

LeNet 诞生于 1994 年，是最早的卷积神经网络之一，并且推动了深度学习领域的发展。自从 1988 年开始，在许多次成功的迭代后，这项由 Yann LeCun 完成的开拓性成果被命名为 LeNet5。LeNet5 的架构基于这样的观点：（尤其是）图像的特征分布在整张图像上，以及带有可学习参数的卷积是一种用少量参数在多个位置上提取相似特征的有效方式。在那时候，没有 GPU 帮助训练，甚至 CPU 的速度也很慢。因此，能够保存参数以及计算过程是一个关键进展。这和将每个像素用作一个大型多层神经网络的单独输入相反。LeNet5 阐述了那些像素不应该被使用在第一层，因为图像具有很强的空间相关性，而使用图像中独立的像素作为不同的输入特征则利用不到这些相关性。

<div align=center><img width=90% height=90% src="/image/2-1.png" alt="LeNet5网络架构"/></div>

<div align=center><img width=90% height=90% src="/image/2-4.png" alt="LeNet5网络架构"/></div>

LeNet5 一共有7层，不包括输入层(32x32):层C1是卷积层，层S2是池化层，层C3是卷积层，层S4是池化层，层C5是卷积层，层F6是全连接层，最后是输出层(10个类别)。LeNet的网络结构虽然简单，但是五脏俱全，卷积层、池化层、全连接层，这些都是现代CNN网络的基本组件。

发展瓶颈：LeNet5的设计较为简单，因此其处理复杂数据的能力有限；近来的研究发现，全连接层的计算代价过大，使用全卷积层组成的神经网络成为趋势。

如今各大深度学习框架中所使用的LeNet5和原始的有些许不同，比如把激活函数改为了现在很常用的ReLU，把平均池化改为最大池化；LeNet5跟现有的conv->pool->ReLU的套路不同，它使用的方式是conv1->pool->conv2->pool2再接全连接层，但是，卷积层后紧接池化层的模式依旧不变。具体如下图所示：

<div align=center><img width=40% height=40% src="/image/2-6.png" alt="LeNet5新网络架构"/></div>

### 2. AlexNet(2012)[2]——王者归来

AlexNet是由2012年发表在NIPS上的一篇文章中提出的，由神经网络的三巨头(Hinton, Lecun, Bengio)之一Hinton的学生Alex提出，这也是深度CNN网络首次应用于图像分类，该方案获得了ILSVRC-2012的Classification任务的冠军，在top-5错误率上达到了15.3%，远超第二名的26.2%。

<div align=center><img width=90% height=90% src="/image/2-2.png" alt="AlexNet网络架构"/></div>

<div align=center><img width=70% height=70% src="/image/2-3.png" alt="AlexNet网络架构"/></div>

AlexNet网络一共包含8层，包括5个卷积层和3个全连接层，具体来说，输入尺寸为227x227x3，有5个卷积层，3个池化层，2个全连接层和1个输出层，输出1000个类别的概率。这里取其前两层进行详细说明。

  * AlexNet共包含5层卷积层和3层全连接层，层数比LeNet多了不少，但卷积神经网络总的流程并没有变化，只是在深度上加了不少；
  - AlexNet针对的是1000类的分类问题，输入图片规定是256×256的三通道彩色图片，为了增强模型的泛化能力，避免过拟合，作者使用了随机裁剪的思路对原来256×256x3的图像进行随机裁剪，得到尺寸为224×224x3的图像，将此图像padding到227x227x3，输入到网络训练；
  * 因为使用多GPU训练，所以可以看到第一层卷积层后有两个完全一样的分支，以加速训练；
  
AlexNet将LeNet扩展为更大的神经网络，可用于学习更复杂的对象和对象层次结构。AlexNet的成功主要归功于以下三个方面的原因：
  * 大量数据，Deep Learning领域应该感谢李飞飞团队贡献出来庞大标注数据集ImageNet；
  - GPU，这种高度并行的计算神器确实助了洪荒之力，没有神器在手，Alex估计不敢搞太复杂的模型；
  * 算法的改进，包括网络变深、数据增强、ReLU、Dropout等。
  
对于AlexNet网络需要注意以下几点：
  * ReLU Nonlinearity：作者首次使用ReLU激活函数代替传统的S型激活函数(tanh, sigmoid)，实验表明，ReLU激活函数可以避免因为神经元进入饱和区域而导致的神经元死亡，并且由于在正半轴梯度始终为1，可以减弱梯度消失问题，已**成为现在深度神经网络的标配**；
  - Local Response Normalization：作者提出了LRN局部响应归一化操作，以提高网络的泛化性能，简单来说，该操作就是对一个feature map沿channel方向的归一化操作，**在目前的深度神经网络中，常常被更好用的Batch Normalization代替**；
  * Overlapping Pooling：对传统的Pooling方法，通常是步长等于池化核大小，即对于同一池化核，池化过程中没有交差重叠。作者这里提出了Overlapping Pooling方法，步长小于池化核，使一次池化过程产生一定的重叠，作者通过实验觉得这对克服过拟合有一定的作用，不过**目前这种操作使用的较少**。
  - Data Augmentation：在Training阶段，作者主要使用了两种数据增强的方法，一种是对图像进行图像翻转、水平镜像和随机裁剪以增加训练数据，另一种是对图像像素使用PCA方法；**第一种方法好像目前用的比较多，第二种较少**；在Testing阶段，作者从一幅图像中裁剪出10个patches进行评估(四个角+中心，水平翻转后重复)，最终的结果是10个patches求均值；
  * Dropout：作者提出了Dropout方法(Dropout方法是Hinton于2012年在Improving neural Networks by preventing co-adaptation of feature detectors这篇文章中提出的，Alex是共同作者)，该方法来源于多模型联合的启发。作者提出，在训练时，以50%的概率将隐含层的神经元输出置零，每一次操作就相当于一个新的模型，并且该操作能够迫使网络学习更加鲁棒的特征。在AlexNet中，作者在前两层全连接层中使用了Dropout操作，**目前该操作已被更好用的Batch Normalization代替**。

### 3. VGGNet(2014)[3]——越走越深

VGGNet是在2015年发表在ICLR上的一篇文章中提出的，网络深度提高到了16至19层，在ILSVRC-2014中，基于该网络的方案获得了Localisation任务的第一名和Classification任务的第二名(同年Classification任务第一名为Inception v1方案)。VGGNet出现前，网络最深也只在10层以内，此时的大多数工作是一方面是使用更小的卷积核和卷积步长，另一方面是使用更大的图像和多尺度对图像进行训练，本文中作者则从另一个角度研究CNN网络的特性–Depth。

<div align=center><img width=60% height=60% src="/image/2-5.jpg" alt="VGGNet网络架构"/></div>

上述表格描述的是VGGNet的网络结构以及诞生过程。为了解决初始化（权重初始化）等问题，VGG采用的是一种Pre-Training的方式，这种方式在经典的神经网络中经常见得到，就是先训练一部分小网络，然后再确保这部分网络稳定之后，再在这基础上逐渐加深。上述表格从左到右体现的就是这个过程，并且当网络处于D阶段的时候，效果是最优的，因此D阶段的网络也就是VGG-16了！E阶段得到的网络就是VGG-19，它也是现在用的比较多的一个预训练模型，在当时算是比较深的网络！VGG-16的16指的是conv+fc的总层数是16，是不包括max pool的层数。下图就是VGG-16的网络结构。

<div align=center><img width=60% height=60% src="/image/2-5.png" alt="VGGNet网络架构"/></div>

由上图看出，VGG-16的结构非常整洁，深度较AlexNet深得多，里面包含多个conv->conv->max_pool这类的结构,VGG的**卷积层都是same的卷积**，即卷积过后的输出图像的尺寸与输入是一致的，它的下采样完全是由max pooling来实现。

VGGNet的闪光点是**卷积层使用更小的滤波器尺寸和间隔**。与AlexNet相比，可以看出VGGNet的卷积核尺寸还是很小的，比如AlexNet第一层的卷积层用到的卷积核尺寸就是11x11，这是一个很大卷积核了。而反观VGGNet，用到的卷积核的尺寸无非都是1×1和3×3的小卷积核，可以替代大的滤波器尺寸。使用小卷积核的好处是：

  * 连续多个3x3的卷积核代替AlexNet中的较大卷积核（11x11，7x7，5x5），对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核是优于采用大的卷积核，因为多层非线性层可以增加网络深度来保证学习更复杂的模式；同时有更少的参数——假设卷积层的输入和输出的特征图大小相同为C，那么三个3×3的卷积层参数个数3×（3×3×C×C）=27CC；一个7×7的卷积层参数为49CC；所以可以把三个3×3滤波器看成是一个7×7滤波器的分解（中间层有非线性的分解）；
  - 1x1卷积核的作用是在不影响输入输出维数的情况下，对输入进行线性形变，然后通过ReLU进行非线性处理，增加网络的非线性表达能力。
  
  因此，由VGGNet我们可以得出结论，**较小的卷积核和较深的网络结构可以提高模型精度**，其优点可以总结为：
  
  * VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3x3）和最大池化尺寸（2x2）；
  - 连续几个小滤波器（3x3）卷积层的组合比一个大滤波器（5x5或7x7）卷积层好；
  * 验证了通过不断加深网络结构可以提升性能。
  
  但 VGGNet 唯一存在的**不足是VGG耗费更多计算资源**，并且使用了更多的参数（这里不是3x3卷积的锅），导致更多的内存占用（140M）。其中绝大多数的参数都是来自于第一个全连接层，并且VGGNet有3个全连接层！巨大的参数空间导致训练一个VGG模型通常要花费更长的时间，所幸有公开的pretrained model让我们很方便的使用。

### 4. NIN(2013)[4]——增强卷积模块功能

NIN(Network In Network)是NUS(National University of Singapore)于2014年发表在ICLR上的一篇文章中提出的，作者首先分析了传统的CNN网络的一些问题，并针对这些问题，提出了自己的改进方法，并将网络结构命名为NIN。在NIN的基础上，Google于2014年提出了GoogLeNet（Inception V1），并随后改进出Inception V3和V4。

作者分析传统的CNN网络存在的问题主要为以下两点：
  * **传统卷积模块非线性能力有限**：对传统的CNN网络，没有经过非线性激活函数之前的卷积操作，实际上只是一个线性操作，如果卷积结果为正，这样经过一个ReLU函数后没有影响，就相当于是一个线性卷积(对前一层receptive field的线性编码)。使用多层这样的线性卷积+ReLU模块虽然在一定程度上可以弥补网络线性抽象能力差的问题，但这样会给下一层网络带来较大的输入，网络的整体参数数量就会增加。
  - **全连接层参数多，容易过拟合**：对传统的CNN网络分类器结构通常是：卷积层作为特征提取器，最后一层卷积输出的结果展开为向量，连接至全连接层进行分类，全连接层的输出再输入到softmax层作为最终的分类结果。作者认为这样的操作可解释性不强，并且全连接层容易引起网络的过拟合。
  
针对以上两个方面的问题，作者提出NIN结构，具有以下两个方面的创新点：
  * **MLP Convolution Layers(多层感知卷积层)**: 使用 Conv+MLP 代替传统卷积层，增强网络提取抽象特征和泛化的能力。由于传统的多层感知机网络具有较强的非线性抽象能力，所以作者考虑将经过卷积操作得到的Feature Map输入到一个MLP网络中，以提高其非线性抽象能力，如下图所示:
  
<div align=center><img width=70% height=70% src="/image/2-7.png" alt="Mlpconv层"/></div>

在多通道的Feature Map上进行该操作，就相当于是将各通道的信息进行混合，此时每个全连接层就相当于对Feature Map进行1x1的卷积操作。作者把这样的conv+1x1conv+1x1conv操作封装成一个子块，由这样的子块堆叠而成的网络就是NIN网络，如下图所示：

<div align=center><img width=50% height=50% src="/image/2-8.png" alt="Mlpconv层和NIN网络"/></div>

从目前的应用来看，1x1 Convolution效果确实非常好，此后的GoogLeNet和ResNet都借鉴了这种操作。 1x1 Convolution好用的原因应该有以下两点：(1)融合Feature Map各通道的信息，提高了网络的抽象能力，进而提高了网络的泛化性能；(2)可以实现对Feature Map层的压缩，以降低参数数量，进而可以提高网络层数；

  * **Global Average Pooling(全局平均池化层)**: 使用平均池化代替全连接层，很大程度上减少参数空间，便于加深网络和训练，有效防止过拟合。对最后一个卷积层的输出，作者提出了一种全局均值池化的方法，各通道的值直接求均值后，以该值作为softmax层的输入以进行分类，具体如下图所示：

<div align=center><img width=60% height=60% src="/image/2-9.jpg" alt="GVP层"/></div>

**Global Average Pooling**的优点如下：(1)不引入新的参数，避免了全连接层带来的参数数量增加和过拟合；(2)增加网络的可解释性，输出的每个通道对应于一个类别；(3)通过实验发现，全局均值池化还有正则化的作用。

### 5. GoogLeNet and Inception(2014)[5]——大浪推手

Inception v1模型是由2015年发表在CVPR上的Going Deeper with Convolutions[5]文章中提出的，文章中首次提出了Inception结构，并把由该结构组成的网络称为GoogLeNet，该网络获得了ILSVRC-2014的Classification任务的冠军。GoogLeNet达到了22层，在当时应该是最深的网络，由于精心设计的网络结构，其参数数量只有AlexNet的8层网络的1/12，约为500w，并且要比AlexNet更为精确。GoogLeNet网络结构如下图所示：

<div align=center><img width=100% height=100% src="/image/2-9.png" alt="GoogLeNet网络架构"/></div>

作者提出了Inception module结构，它借鉴了NIN[4]的一些思想，网络结构相比于传统的CNN结构有很大改变。网络中大量使用1x1的卷积核，NIN中使用1x1卷积核的目的主要是提高网络的非线性能力，而这里作者用它的主要目的是进行降维，**将参数空间进行压缩，去除掉无用的稀疏数据，使参数空间更为稠密**，这样可减少参数数量，进而可增加网络的深度和宽度。

根据感受野的概念，较高层的feature map中一个像素点对应于原图的一个区域的信息，所以作者考虑分别使用1x1, 3x3, 5x5的卷积核分别对前一层的feature map进行卷积，以覆盖图像中不同大小的物体，通过padding操作可使输出的feature map的shape相同，然后在channel方向联合这些feature map以作为当前层的输出。不同大小的卷积核相当于对原图像进行多尺度的处理，后一层在处理时，相当于同时处理了不同尺度图像的信息。在较高层时特征更为抽象，需要用较大的卷积核来融合不同的特征，所以在较高层时，3x3, 5x5卷积核的数量要多一点。具体的Inception结构如下图所示：

<div align=center><img width=60% height=60% src="/image/2-10-1.png" alt="Inception module原始版本"/></div>
