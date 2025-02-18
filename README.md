# Research on Perception Module for Autonomous Driving

本人刚刚从事无人驾驶中感知模块的研究，目前主要研究采用多传感器融合进行3D目标检测，现将研究过程中研读的论文采用中英文对照翻译的形式整理在此仓库中。

****
## 目录
* [CNN](#cnn)
  * [1 LeNet5(1998)——开山之作](#1-lenet5)
  * [2 AlexNet(2012)——王者归来](#2-alexnet)
  * [3 VGGNet(2014)——越走越深](#3-vggnet)
  * [4 NIN(2013)——增强卷积模块功能](#4-nin)
  * [5 GoogLeNet and Inception(2014)——大浪推手](#5-googlenet)
  * [6 ResNet(2015)——里程碑式创新](#6-resnet)
  * [7 DenseNet(2017)——既往开来](#7-densenet)
  * [8 Reference](#8-reference)
* [2D Object Detection](#2d-object-detection)
  * [Two-Stage目标检测方法](#two-stage目标检测方法)
    * [1 RCNN(2014)](#1-rcnn)
    * [2 SPPNet(2014)](#2-sppnet)
    * [3 Fast RCNN(2015)](#3-fast-rcnn)
    * [4 Faster RCNN(2015)](#4-faster-rcnn)
    * [5 Pyramid Networks(2017)](#5-pyramid-networks)
  * [One-Stage目标检测方法](#one-stage目标检测方法)
    * [1 YOLO(2016-2017-2018)](#1-yolo)
    * [2 SSD(2016)](#2-ssd)
    * [3 Retina-Net(2017)](#3-retina-net)
  * [Reference](#reference)
* [3D Object Detection](#3d-object-detection)
  * [1 基于视觉图像方法](#1-基于视觉图像方法)
  * [2 基于激光雷达点云方法](#2-基于激光雷达点云方法)
  * [3 基于多模态信息融合方法](#3-基于多模态信息融合方法)
  * [4 Reference](#4-reference)

****
## CNN

卷积神经网络(Convolutional Neural Network, CNN)是深度学习领域重要的神经网络框架之一，也在计算机视觉领域中发挥着重要的作用。CNN从90年代的LeNet开始，21世纪初沉寂了10年，其锋芒被SVM等手工设计的特征盖过，直到12年AlexNet开始又再焕发第二春，从ZF Net到VGG，GoogLeNet再到ResNet和DenseNet，网络越来越深，架构越来越复杂，解决反向传播时梯度消失的方法也越来越巧妙。下图显示了CNN的发展脉络。

<div align=center><img width=90% height=90% src="/image/2-0.png" alt="CNN发展脉络"/></div>

下图直观显示了经典的CNN网络的性能对比，其中，纵轴表示在ImageNet图像分类上的Top-1正确率，横轴表示计算量，图中圆圈的大小与网络所需的计算量成正比：

<div align=center><img width=70% height=70% src="/image/2-14.png" alt="经典CNN网络性能对比"/></div>

表2-1总结了经典CNN网络的贡献和其发展空间：

<div align=center><img width=90% height=90% src="/image/2-13.png" alt="经典CNN网络贡献和发展空间"/></div>

### 1 LeNet5

LeNet[1]诞生于 1994 年，是最早的卷积神经网络之一，并且推动了深度学习领域的发展。自从 1988 年开始，在许多次成功的迭代后，这项由 Yann LeCun 完成的开拓性成果被命名为 LeNet5。LeNet5 的架构基于这样的观点：（尤其是）图像的特征分布在整张图像上，以及带有可学习参数的卷积是一种用少量参数在多个位置上提取相似特征的有效方式。在那时候，没有 GPU 帮助训练，甚至 CPU 的速度也很慢。因此，能够保存参数以及计算过程是一个关键进展。这和将每个像素用作一个大型多层神经网络的单独输入相反。LeNet5 阐述了那些像素不应该被使用在第一层，因为图像具有很强的空间相关性，而使用图像中独立的像素作为不同的输入特征则利用不到这些相关性。

<div align=center><img width=90% height=90% src="/image/2-1.png" alt="LeNet5网络架构"/></div>

<div align=center><img width=100% height=100% src="/image/2-4.png" alt="LeNet5网络架构"/></div>

LeNet5 一共有7层，不包括输入层(32x32):层C1是卷积层，层S2是池化层，层C3是卷积层，层S4是池化层，层C5是卷积层，层F6是全连接层，最后是输出层(10个类别)。LeNet的网络结构虽然简单，但是五脏俱全，卷积层、池化层、全连接层，这些都是现代CNN网络的基本组件。

发展瓶颈：LeNet5的设计较为简单，因此其处理复杂数据的能力有限；近来的研究发现，全连接层的计算代价过大，使用全卷积层组成的神经网络成为趋势。

如今各大深度学习框架中所使用的LeNet5和原始的有些许不同，比如把激活函数改为了现在很常用的ReLU，把平均池化改为最大池化；LeNet5跟现有的conv->ReLU->pool的套路不同，它使用的方式是conv1->pool->conv2->pool2再接全连接层，但是，卷积层后紧接池化层的模式依旧不变。pytorch实现leNet5的代码如下：

```
import torch.nn as nn
class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = LeNet5()
print(net)
```
代码的输出结果为：

```
LeNet5(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

### 2 AlexNet

AlexNet[2]是由2012年发表在NIPS上的一篇文章中提出的，由神经网络的三巨头(Hinton, Lecun, Bengio)之一Hinton的学生Alex提出，这也是深度CNN网络首次应用于图像分类，该方案获得了ILSVRC-2012的Classification任务的冠军，在top-5错误率上达到了15.3%，远超第二名的26.2%。

<div align=center><img width=90% height=90% src="/image/2-2.png" alt="AlexNet网络架构"/></div>

<div align=center><img width=50% height=50% src="/image/2-3.png" alt="AlexNet网络架构"/></div>

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
  
torchvision.models模块的子模块中包含AlexNet网络模型，使用如下代码直接调用，源代码参见：[Alexnet](https://pytorch.org/docs/stable/_modules/torchvision/models/alexnet.html#alexnet "Alexnet")

```
import torchvision
model = torchvision.models.alexnet(pretrained=False)
print(model)
```
输出结果为：
```
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace)
    (3): Dropout(p=0.5)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

### 3 VGGNet

VGGNet[3]是在2015年发表在ICLR上的一篇文章中提出的，网络深度提高到了16至19层，在ILSVRC-2014中，基于该网络的方案获得了Localisation任务的第一名和Classification任务的第二名(同年Classification任务第一名为Inception v1方案)。VGGNet出现前，网络最深也只在10层以内，此时的大多数工作是一方面是使用更小的卷积核和卷积步长，另一方面是使用更大的图像和多尺度对图像进行训练，本文中作者则从另一个角度研究CNN网络的特性–Depth。

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
  
torchvision.models模块的子模块中包含VGG网络模型，具体涵盖了VGG11,VGG13,VGG16,VGG19，这里使用如下代码调用常用的VGG16，源代码参见：[VGG](https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg16 "VGG16")

```
import torchvision
model = torchvision.models.vgg16(pretrained=False)
print(model)
```
输出结果为：
```
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace)
    (2): Dropout(p=0.5)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace)
    (5): Dropout(p=0.5)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

### 4 NIN

NIN(Network In Network)[4]是NUS(National University of Singapore)于2014年发表在ICLR上的一篇文章中提出的，作者首先分析了传统的CNN网络的一些问题，并针对这些问题，提出了自己的改进方法，并将网络结构命名为NIN。在NIN的基础上，Google于2014年提出了GoogLeNet（Inception V1），并随后改进出Inception V3和V4。

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

### 5 GoogLeNet

Inception v1模型是由2015年发表在CVPR上的Going Deeper with Convolutions[5]文章中提出的，文章中首次提出了Inception结构，并把由该结构组成的网络称为GoogLeNet，该网络获得了ILSVRC-2014的Classification任务的冠军。GoogLeNet达到了22层，在当时应该是最深的网络，由于精心设计的网络结构，其参数数量只有AlexNet的8层网络的1/12，约为500w，并且要比AlexNet更为精确。GoogLeNet网络结构如下图所示：

<div align=center><img width=50% height=50% src="/image/2-9.png" alt="GoogLeNet网络架构"/></div>

作者提出了Inception module结构，它借鉴了NIN[4]的一些思想，网络结构相比于传统的CNN结构有很大改变。网络中大量使用1x1的卷积核，NIN中使用1x1卷积核的目的主要是提高网络的非线性能力，而这里作者用它的主要目的是进行降维，**将参数空间进行压缩，去除掉无用的稀疏数据，使参数空间更为稠密**，这样可减少参数数量，进而可增加网络的深度和宽度。

由于图像信息位置的巨大差异，为卷积操作选择合适的卷积核大小就比较困难。信息分布更全局性的图像偏好较大的卷积核，信息分布比较局部的图像偏好较小的卷积核。作者考虑能否在同一层级上运行具备多个尺寸的滤波器，以此**替代人工确定该层卷积层中滤波器的类型或者是否创建卷积层和池化层，让网络自行学习该层的网络结构**，由此提出了Inception module结构，其分别使用1x1, 3x3, 5x5的卷积核分别对前一层的feature map进行卷积，以覆盖图像中不同大小的物体，通过padding操作可使输出的feature map的shape相同，然后在channel方向联合这些feature map以作为当前层的输出。具体的Inception结构如下图所示：

<div align=center><img width=60% height=60% src="/image/2-10-1.png" alt="Inception module原始版本"/></div>

但是这里有一个问题：在较高层时，channel数目较多，5x5卷积操作所带来的计算量非常大，特别是叠加pooling操作后(pooling操作channel数量不变)，channel数量会变得非常多。这里作者提出了第二个版本的Inception结构，在对输入的feature map进行卷积操作前，先使用1x1的卷积对特征进行压缩，之后的卷积操作相当于又将稠密的特征稀疏化。而在pooling操作时，则是后进行1x1卷积操作，以减少channel数量。

<div align=center><img width=65% height=65% src="/image/2-10-2.png" alt="Inception module改进版本"/></div>

GoogLeNet便是应用上述Inception结构所构成的网络，只算有训练参数的层的情况下，网络有22层，网络结构可参考论文中的图，具体每层的参数如下表：

<div align=center><img width=80% height=80% src="/image/2-10-3.png" alt="GoogLeNet网络结构和参数表"/></div>

表格中需要注意以下几点：

  * 表格中的#3x3 reduce和#5x5 reduce一栏表示在3x3和5x5卷积前所用的1x1卷积核的数量；
  - 表格中的inception(-a/b...)是对feature map大小相同的情况下对使用的Inception模块的编号；
  * Inception结构中的max pooling操作使用的是3x3的步长为1的池化，而用于Inception之间的则是3x3的步长为2的池化，以缩小feature map的体积；
  - 网络中使用average pooling代替全连接层用于分类，这给网络性能带来了一定的提升；注意这里的average pooling不同于全局均值池化，此处average pooling后，又对结果进行了线性组合(FC)后才形成最终的分类结果；
  * 虽然没有使用全连接层，但是网络中依然使用了dropout层，作者认为这很有必要；
  - 网络中也使用了LRN；
  * 网络在inference时的计算量约为1.5 billion multiply-adds；
  
由于网络层数较深，所以会带来梯度消失的问题。为了应对该问题，在训练阶段，作者为网络添加了辅助分类器，即使用网络中间某层的feature map进行分类，计算得到的loss以一定权重添加到总的loss中用于训练，在测试阶段则丢弃这些辅助分类器。GoogLeNet网络分别在inception(4a)和inception(4d)的顶部添加了辅助分类器，其loss按0.3的权重添加到总的loss中。 辅助分类器的结构参考论文。

torchvision.models模块的子模块中包含GoogLeNet网络模型，使用如下代码直接调用，源代码参见：[GoogLeNet](https://pytorch.org/docs/stable/_modules/torchvision/models/googlenet.html#googlenet "GoogLeNet")

```
import torchvision
model = torchvision.models.googlenet(pretrained=False)
print(model)
```
输出结果为：
```
GoogLeNet(
  (conv1): BasicConv2d(
    (conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  )
  (maxpool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
  (conv2): BasicConv2d(
    (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv3): BasicConv2d(
    (conv): Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  )
  (maxpool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
  (inception3a): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (inception3b): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(32, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (maxpool3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
  (inception4a): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(480, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(208, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(16, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (inception4b): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (inception4c): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (inception4d): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(288, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (inception4e): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(528, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (maxpool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
  (inception5a): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (inception5b): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(832, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(48, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (aux1): InceptionAux(
    (conv): BasicConv2d(
      (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (fc1): Linear(in_features=2048, out_features=1024, bias=True)
    (fc2): Linear(in_features=1024, out_features=1000, bias=True)
  )
  (aux2): InceptionAux(
    (conv): BasicConv2d(
      (conv): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (fc1): Linear(in_features=2048, out_features=1024, bias=True)
    (fc2): Linear(in_features=1024, out_features=1000, bias=True)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (dropout): Dropout(p=0.2)
  (fc): Linear(in_features=1024, out_features=1000, bias=True)
)
```

**Inception v2** 和 **Inception v3** 来自同一篇论文[8]，作者积极探索扩展网络的方法，旨在通过适当的分解卷积与积极的正则化尽可能高效地计算。

**Inception v2**主要是运用**Batch Normalization**改进Inception。Batch Normalization出现后基本取代了局部响应归一化操作(LRN)，目前已经成为CNN网络的标准配置。BN操作通常用于卷积层和激活层之间，用于对各层的feature map进行归一化操作。BN的主要作用就是加速网络训练和防止梯度消失。

**Inception v3**中作者给出了一种分解Inception结构中较大卷积核的方法，可降低参数数量。有两种分解方法：

  * 分解为对称的小卷积核；
  - 分解为不对称的卷积核；

第一种方法和VGG的思想很像，如下图所示，对左图Inception V1中的1个5x5的卷积核替换成右图的2个3x3的卷积核:

<div align=center><img width=80% height=80% src="/image/2-10-4.png" alt="InceptionV3(A)"/></div>

第二种方法就是将nxn的卷积核替换成 1xn 和 nx1 的卷积核堆叠。有两种堆叠方法：一种是沿深度方向的堆叠，如下图的左图所示；另一种是沿宽度方向堆叠，如下图的右图所示。第二种方法在大维度的特征图上表现不好，在特征图维度为12-20上表现好。

<div align=center><img width=70% height=70% src="/image/2-10-5.png" alt="InceptionV3(B&C)"/></div>

前面三个原则用来构建三种不同类型的 Inception 模块（这里我们按引入顺序称之为模块 A、B、C，这里使用「A、B、C」作为名称只是为了清晰期间，并不是它们的正式名称）。架构如下所示：

<div align=center><img width=50% height=50% src="/image/2-10-6.png" alt="InceptionV3网络架构"/></div>

这里，「figure 5」是模块 A，「figure 6」是模块 B，「figure 7」是模块 C。

需要注意的是，CNN中一般使用Pooling操作来降低feature map的size，但是作者认为直接对feature map进行Pooling操作会遇到前面所述的表达瓶颈的问题。作者提出了一种新的方法来解决此问题，即同时进行步长为2的卷积核池化操作，再将其feature map进行联合，如下图所示，这也是作者在网络中采用的Pooling方案：

<div align=center><img width=40% height=40% src="/image/2-10-7.png" alt="Pooling方案"/></div>

同Inception V1一样，Inception V3中也使用辅助分类器，即除了主分类器之外，还在网络结构中的某一层，论文中为17x17x768那一层，添加了一个分支用来做辅助分类，具体结构如下图所示：

<div align=center><img width=50% height=50% src="/image/2-10-8.png" alt="辅助分类器"/></div>

torchvision.models模块的子模块中包含InceptionV3网络模型，使用如下代码直接调用，源代码参见：[InceptionV3](https://pytorch.org/docs/stable/_modules/torchvision/models/inception.html#inception_v3 "InceptionV3")

```
import torchvision
model = torchvision.models.inception_v3(pretrained=False)
print(model)
```
输出结果为：
```
Inception3(
  (Conv2d_1a_3x3): BasicConv2d(
    (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  )
  (Conv2d_2a_3x3): BasicConv2d(
    (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  )
  (Conv2d_2b_3x3): BasicConv2d(
    (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  )
  (Conv2d_3b_1x1): BasicConv2d(
    (conv): Conv2d(64, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  )
  (Conv2d_4a_3x3): BasicConv2d(
    (conv): Conv2d(80, 192, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  )
  (Mixed_5b): InceptionA(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch5x5_1): BasicConv2d(
      (conv): Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch5x5_2): BasicConv2d(
      (conv): Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_1): BasicConv2d(
      (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_2): BasicConv2d(
      (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_3): BasicConv2d(
      (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch_pool): BasicConv2d(
      (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (Mixed_5c): InceptionA(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch5x5_1): BasicConv2d(
      (conv): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch5x5_2): BasicConv2d(
      (conv): Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_1): BasicConv2d(
      (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_2): BasicConv2d(
      (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_3): BasicConv2d(
      (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch_pool): BasicConv2d(
      (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (Mixed_5d): InceptionA(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch5x5_1): BasicConv2d(
      (conv): Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch5x5_2): BasicConv2d(
      (conv): Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_1): BasicConv2d(
      (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_2): BasicConv2d(
      (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_3): BasicConv2d(
      (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch_pool): BasicConv2d(
      (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (Mixed_6a): InceptionB(
    (branch3x3): BasicConv2d(
      (conv): Conv2d(288, 384, kernel_size=(3, 3), stride=(2, 2), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_1): BasicConv2d(
      (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_2): BasicConv2d(
      (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_3): BasicConv2d(
      (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), bias=False)
      (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (Mixed_6b): InceptionC(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7_1): BasicConv2d(
      (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7_2): BasicConv2d(
      (conv): Conv2d(128, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7_3): BasicConv2d(
      (conv): Conv2d(128, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_1): BasicConv2d(
      (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_2): BasicConv2d(
      (conv): Conv2d(128, 128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_3): BasicConv2d(
      (conv): Conv2d(128, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_4): BasicConv2d(
      (conv): Conv2d(128, 128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_5): BasicConv2d(
      (conv): Conv2d(128, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch_pool): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (Mixed_6c): InceptionC(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7_1): BasicConv2d(
      (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7_2): BasicConv2d(
      (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7_3): BasicConv2d(
      (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_1): BasicConv2d(
      (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_2): BasicConv2d(
      (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_3): BasicConv2d(
      (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_4): BasicConv2d(
      (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_5): BasicConv2d(
      (conv): Conv2d(160, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch_pool): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (Mixed_6d): InceptionC(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7_1): BasicConv2d(
      (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7_2): BasicConv2d(
      (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7_3): BasicConv2d(
      (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_1): BasicConv2d(
      (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_2): BasicConv2d(
      (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_3): BasicConv2d(
      (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_4): BasicConv2d(
      (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_5): BasicConv2d(
      (conv): Conv2d(160, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch_pool): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (Mixed_6e): InceptionC(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7_1): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7_2): BasicConv2d(
      (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7_3): BasicConv2d(
      (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_1): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_2): BasicConv2d(
      (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_3): BasicConv2d(
      (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_4): BasicConv2d(
      (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7dbl_5): BasicConv2d(
      (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch_pool): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (AuxLogits): InceptionAux(
    (conv0): BasicConv2d(
      (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (conv1): BasicConv2d(
      (conv): Conv2d(128, 768, kernel_size=(5, 5), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(768, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (fc): Linear(in_features=768, out_features=1000, bias=True)
  )
  (Mixed_7a): InceptionD(
    (branch3x3_1): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3_2): BasicConv2d(
      (conv): Conv2d(192, 320, kernel_size=(3, 3), stride=(2, 2), bias=False)
      (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7x3_1): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7x3_2): BasicConv2d(
      (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7x3_3): BasicConv2d(
      (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch7x7x3_4): BasicConv2d(
      (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (Mixed_7b): InceptionE(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3_1): BasicConv2d(
      (conv): Conv2d(1280, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3_2a): BasicConv2d(
      (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3_2b): BasicConv2d(
      (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_1): BasicConv2d(
      (conv): Conv2d(1280, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(448, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_2): BasicConv2d(
      (conv): Conv2d(448, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_3a): BasicConv2d(
      (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_3b): BasicConv2d(
      (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch_pool): BasicConv2d(
      (conv): Conv2d(1280, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (Mixed_7c): InceptionE(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(2048, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3_1): BasicConv2d(
      (conv): Conv2d(2048, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3_2a): BasicConv2d(
      (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3_2b): BasicConv2d(
      (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_1): BasicConv2d(
      (conv): Conv2d(2048, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(448, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_2): BasicConv2d(
      (conv): Conv2d(448, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_3a): BasicConv2d(
      (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_3b): BasicConv2d(
      (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch_pool): BasicConv2d(
      (conv): Conv2d(2048, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (fc): Linear(in_features=2048, out_features=1000, bias=True)
)
```

### 6 ResNet

ResNet v1是由Kaiming He于2016年发表在CVPR上的文章Deep Residual Learning[6]中提出的，残差学习的方法有效地解决了随着网络深度增加而造成的性能退化的问题，在这篇文章中，最深的网络深度达到了152层。以该结构为基础的所构建的网络，赢得了ILSVRC-2015的Classilfication/Localization/Detection任务的冠军，同时赢得了COCO-2015的Detection/Segmentation任务的冠军，并且在COCO目标检测数据集上，获得了相对28%的提升。

目前的研究已经表明，提高网络深度可以提高网络性能，但网络较深则会带来**梯度消失**和**梯度爆炸**的问题，目前该问题可以通过**Normalized Initialization**和**Batch Normalization**很好地解决。但通过实验发现，不断增加网络深度还会带来**网络退化**的问题，较深网络的训练和测试误差都比较浅层网络大，造成这一现象的不是过拟合，而是**网络退化**。作者对这一问题进行了分析，提出了残差网络的概念。

如果假设非线性的卷积网络层可以近似复杂的函数，那么它应该也可以去近似一个**恒等映射**，即:输入x经过非线性卷积层后的输出仍为x。基于以上假设，在一个已经获得较好性能的浅层网络上继续叠加新的卷积层，这些新加的卷积层都被构造为**恒等映射**，其他层直接是对原有网络的复制，那么原则上讲，新构造的较深的网络的误差应该不会比原有网络大。然而实验却发现**网络退化**现象，因此可推断**非线性卷积层在近似恒等映射方面有困难**。

仍然是基于非线性的卷积网络层可以近似复杂的函数这一假设，并设原来经过两层卷积所近似的映射函数为H(x)，则原则上这两层卷积同样可近似H(x)-x这一残差函数，如下图所示，图中的F(x)即为两个卷积层所要映射的残差函数H(x)-x，则将输入x直接连到输出后，可得到网络的新的输出F(x)+x，对于正常的卷积层(比如上述例子中的浅层网络中的卷积层)，该模块的最终映射为F(x)+x=H(x)-x+x=H(x)，不受影响；而对于要实现恒等映射的卷积层来说，新的残差结构使得卷积层只需要实现F(x)=H(x)-x=0即可，后面的实验表明，**非线性卷积层去近似该残差函数要比去近似一个恒等映射的效果要好的多**。

<div align=center><img width=50% height=50% src="/image/2-11-1.png" alt="ResNet的残差学习模块"/></div>

对于以上残差模块，要注意以下几点：

  * 输入x与输出的残差连接是element-wise addition，即对应元素相加，所以要保证输入与输出的shape相同；
  - 对该模块，经过残差连接相加后，才经过最后的非线性激活函数；
  * 论文中单个模块中包含的卷积层的个数为2-3层，如下图所示，论文中较浅层的网络应用左侧模块，较深层的网络应用右侧模块，右侧模块前后两个1x1的卷积层分别达到降低和提高维度的作用，可用于构建较深的网络。
  
<div align=center><img width=80% height=80% src="/image/2-11-2.png" alt="两层和三层残差学习模块"/></div>

ResNet有不同的网络层数，比较常用的是50-layer，101-layer，152-layer。他们都是由上述的残差模块堆叠在一起实现的。具体的网络配置如下表所示：

<div align=center><img width=80% height=80% src="/image/2-11-3.png" alt="ResNet不同层数时的网络配置"/></div>

如下图是经典的resnet18的网络结构：

<div align=center><img width=80% height=80% src="/image/2-11-4.jpg" alt="ResNet18网络结构"/></div>

torchvision.models模块的子模块中包含ResNet网络模型，如下图显示调用ResNet18，源代码参见：[ResNet](https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnext50_32x4d "ResNet")

```
import torchvision
model = torchvision.models.resnet18(pretrained=False)
print(model)
```
输出结果为：
```
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
```

### 7 DenseNet

作为CVPR2017年的Best Paper, DenseNet[7]脱离了加深网络层数(ResNet[6])和加宽网络结构(Inception[5])来提升网络性能的定式思维，从特征的角度考虑，通过特征重用和旁路(Bypass)设置，提出了一种稠密残差连接的模块，并据此构建出较深的网络结构，取得了不错的效果。如下图所示，为一个5层稠密残差连接模块的示意图，在图中，第一层X0为输入，其后各层为卷积后的特征图，并且特征图的数目都为4，即该稠密残差连接模块的增长率超参数 k = 4：

<div align=center><img width=70% height=70% src="/image/2-12-1.png" alt="Dense模块结构图"/></div>

该模块的特点为：

  * 各个卷积层的输入为前面各个卷积层输出特征图的叠加(沿channel方向连接)；
  - 叠加之后的特征图输入经过一种预激活结构，即BN-ReLU-ConV，使各卷积层输出的特征图具有同样的shape(w x h x c);
  * 模块最终的输出经过Transition Layer，该层负责对特征图进行降采样，在作者的实验中，该层的配置为BN-Conv(1x1)-AveragePooling(2x2);
  
使用该模块构成的卷积网络的示意图如下，具体的网络配置见下表，网络中，首先使用一个较大的卷积核(如7x7)和步长对图像进行卷积，提取出一个较小尺寸和多通道的特征图，然后叠加Dense Block模块进行特征提取工作，模块之间由Transition Layer进行降采样减小特征的尺寸：

<div align=center><img width=80% height=80% src="/image/2-12-2.png" alt="DenseNet网络结构图"/></div>

<div align=center><img width=80% height=80% src="/image/2-12-3.png" alt="DenseNet网络结构配置表"/></div>

论文中作者对DenseNet网络进行了优化，共经历以下三个阶段:

  * **Basic DenseNet**:前文提到每个Dense Block模块中的各个卷积层之间使用一种预激活的结构，即BN-RelU-Conv，在最初的版本中，其结构如下图所示：
  <div align=center><img width=50% height=50% src="/image/2-12-4.png" alt="Basic DenseNet"/></div>
  
  * **DenseNet-B**:作者对BN-ReLU-Conv结构进行改进，即输入的特征图首先经过Bottleneck Layer(Conv 1x1)降低特征图channel数量，在实验中降维4k，再经过Conv 3x3降维k，经过改善的预激活结构为BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)，如下图所示，具有Bottleneck Layer的DenseNet被作者称为**DenseNet-B**；
  <div align=center><img width=75% height=75% src="/image/2-12-5.png" alt="DenseNet-B"/></div>
  
  * **DenseNet-C**:作者同样在Transition Layer中进行改进，在该层中降低特征图channel的数量，若前一层Dense Block中的channel为m，则可以设置Dense Block间的转移层输出的channel个数为theta x m，0<theta<=1，这样可通过theta参数控制网络规模。论文中将使用该方法并且theta=0.5的DenseNet记为**DenseNet-C**，将综合了DenseNet-B和DenseNet-C的网络称为**DenseNet-BC**。

### 8 Reference

[1] Lécun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11):2278-2324.

[2] Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[C]//Advances in Neural Information Processing Systems(NIPS). 2012: 1097-1105.

[3] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.

[4] Lin M, Chen Q, Yan S. Network in network[J]. arXiv preprint arXiv:1312.4400, 2013.

[5] Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition(CVPR). 2015: 1-9.

[6] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition(CVPR). 2016: 770-778.

[7] Huang G, Liu Z, Van Der Maaten L, et al. Densely connected convolutional networks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition(CVPR). 2017: 4700-4708.

[8] Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the inception architecture for computer vision[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition(CVPR). 2016: 2818-2826.

****
## 2D Object Detection

目标检测(Object Detection)即找出图像中所有感兴趣的物体，包含目标定位和目标分类两个子任务，即需同时确定目标物体的位置和类别。目标检测任务面临的挑战主要是：物体的尺寸变化范围大，物体的姿态角度不定，物体的类别多样，物体可能出现在图片的任何地方......目前学术和工业界出现的目标检测算法分为3类：

  * 传统的目标检测方法：区域选择(采用滑动窗口的穷举策略)+ 特征提取(SIFT/HOG/DPM等) + 分类器分类(SVM/Haar)以及上述方法的诸多改进、优化；
  - 候选区域/框 + 深度学习分类：通过提取候选区域，并对相应区域进行以深度学习方法为主的分类，如R-CNN、SPPNet、Fast RCN、Faster R-CNN、R-FCN等系列方法；
  * 基于深度学习的回归方法：YOLO、SSD等方法。
  
我们重点关注后两种基于深度学习的目标检测方法，其中上述第二种称为**Two-Stage目标检测方法**，即这类检测算法将检测问题划分为两个阶段，第一个阶段首先产生候选区域(Region Proposals)，其包含目标大概的位置信息，然后第二个阶段对候选区域进行分类和位置精修；上述第三种方法称为**One-Stage目标检测方法**，这类检测算法不需要Region Proposal阶段，可以通过一个Stage直接产生物体的类别概率和位置坐标值。目标检测模型的主要性能指标是**检测准确度**和**速度**，其中准确度主要考虑物体的定位以及分类准确度。一般情况下，Two-Stage算法在准确度上有优势，而One-Stage算法在速度上有优势。不过，随着研究的发展，两类算法都在两个方面做改进，均能在准确度以及速度上取得较好的结果。

下图概括了目标检测的路线图[1]，路线图将目标检测的进展划分为两个历史时期：1)传统目标检测期(2014年之前)；2)深度学习检测期(2014年之后)。

<div align=center><img width=80% height=80% src="/image/3-1.png" alt="目标检测的路线图"/></div>

### Two-Stage目标检测方法

#### 1 RCNN



#### 2 SPPNet

#### 3 Fast RCNN

#### 4 Faster RCNN

#### 5 Pyramid Networks

### One-Stage目标检测方法

#### 1 YOLO

#### 2 SSD

#### 3 Retina-Net

### Reference

[1] Zhengxia Zou, Zhenwei Shi, Yuhong Guo, and Jieping Ye. (May. 2019). “Object Detection in 20 Years: A Survey.” [Online]. Available: https://arxiv.org/abs/1905.05055

[2] Girshick R, Donahue J, Darrell T, et al. Rich feature hierarchies for accurate object detection and semantic segmentation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2014: 580-587.

****
## 3D Object Detection

自动驾驶汽车(Autonomous Vehicle, AV)需要准确感知其周围环境才能可靠运行，物体检测是这种感知系统的基本功能[1]。自动驾驶中2D目标检测方法的性能已经得到很大的提高，在KITTI物体检测基准上实现了超过90%的平均精度(Average Precision, AP)。3D目标检测相比于2D目标检测仍然存在很大的性能差距，并且3D目标检测任务对于其它自动驾驶任务至关重要，如路径规划、碰撞避免，因此有必要对3D目标检测方法进行进一步的研究。表1-1显示了2D和3D目标检测的对比：

<div align=center><img width=70% height=70% src="/image/1-1.png" alt="2D和3D目标检测的对比"/></div>

自动驾驶车辆使用的传感器种类繁多，但大多数关于AV的研究都集中在相机和激光雷达上。相机分为被动式的单目相机和立体相机以及主动式的TOF相机。单目相机以像素强度的形式提供详细信息，其优点是以更大的尺度显示目标物体的形状和纹理特性，其缺点是缺少深度信息、且易受光照和天气条件影响；立体相机可用于恢复深度通道，但是匹配算法增加了计算处理负担；TOP相机提供的深度信息是通过测量调制的红外脉冲的飞行时间，优点是计算复杂度较低但缺点是分辨率较低。激光雷达传感器作为有源传感器通过发射激光束并测量从发射到接收的时间得到3D位置，其产生的是一组3D点组成的3D点云(Point Cloud),这些点云通常是稀疏且空间分布不均，优点是在恶劣天气和极端光照条件下仍能可靠运作。目前的价格问题仍是阻碍其大规模民用的主要原因之一。表1-2显示了AV中常用的几种传感器的对比。

<div align=center><img width=70% height=70% src="/image/1-2.png" alt="传感器对比"/></div>

3D目标检测方法根据输入的数据类型分为基于视觉图像方法，基于激光雷达点云方法，基于多模态信息融合方法。其中，基于视觉图像方法主要研究基于单目图像方法，基于激光雷达点云的方法又可分为三个之类，基于投影、体素和PointNet方法。这三种方法的详细对比见表1-3。

<div align=center><img width=90% height=90% src="/image/1-3.png" alt="3D目标检测方法对比"/></div>


### 1 基于视觉图像方法

<div align=center><img width=80% height=80% src="/image/1-4.png" alt="基于单目视觉方法总结"/></div>

### 2 基于激光雷达点云方法


### 3 基于多模态信息融合方法

### 4 Reference

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
