# Data Augmentation in Medical Field
Realization of some of my mentor's strategies on data augmentation in medical field
## Background
医疗领域使用深度学习的一个问题是数据量太少。图像数据的数据增强已有很实用的策略并被广泛使用，但是非图像数据的数据增强还没有。而我们现在有的数据比较特殊，它不是图像数据，但是数据之间仍存在位置上的关系。我们为这类数据设计了几种数据增强策略，并通过实验证明它们是否有效。
## The Data 
数据是人脑MRI扫描结果，一共有34个扫描的区域，被扫描后每个区域会给出一个数值，每个患者也就是一个记录给出TA四年的数据，也就是一共有34 * 4 个features，数据的label为一个分类结果，一共有五种，在这里就标为ABCDE。34个扫描区域还被分为6个lobes，每个lobes包括不同的区域。还有三个额外的feature包括患者性别，设备品牌等。
## Augmentation Strategic
### Combination
最初的策略，是将X个同类别的数据直接组合到一起来生成新的数据。该方法可以轻松生成大量数据，X的数量如果超过4，生成该策略所有可能数据就已经没有可能了。这个方法的问题，我认为是通过改变网络结构，我们能得到同样的效果。
### Kernal
#### Fixed Kernal
选择X * Y大小的kernal，将kernal中选定的区域替换成其他记录同样区域的数据。这种方法同样可以生成大量数据。我们原本数据为一维的长度为134的数据，在kernal替换策略中，我们需先将数据转换成4 * 34的二维数据。在这里我们在不同year之间的数据替换可能能保存疾病的一种变化趋势。
#### Fixed Kernals for Different Lobes
脑科学中将大脑一定区域归为一类，这每一类就称为一个lobe。因此为每一个lobe采用不同的kernal来生成数据显得非常合理，因为在同lobe的区域可能有更多的联系，替换生成的数据更接近该类别
#### Dynamic Kernals for Different Lobes
对应每一种类别的数据生成，采用不同的kernals。这样做的原因主要是为了解决数据的类别不均匀的问题，数据不均衡会导致训练出来的分类器是naive的--预测结果总是为比例最高的类别即可保证训练集的loss很低，同时如果采用是同样分布的测试集，该分类器的准确率Accuracy也会很高，但是一些类别的Percision和Recall会为0，这样的分类器显然不是我们想要的。其解决方式之一即使提供类别均衡的训练数据，而我们在数据增强的过程中正好可以做到这点。
## Code
机器学习框架主要使用TensorFlow
### MCDA-NN.py
文件包含所有框架训练
#### def watch(object)：
主要进行TensorFlow的变量查看
#### def hebing01234()
将数据增强完后的数据根据类别进行合并
#### def shujuload(tt):
to be continued



