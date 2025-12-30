import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Discriminator(nn.Module):
    def __init__(self, input_nc, mid_nc):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            input_nc, mid_nc, kernel_size=4, stride=2, padding=1
        )  # 对输入特征图做二维卷积，同时下采样（尺寸减半）并提取特征
        # 为什么要设这些参数？
        # 输入通道数input_nc，输出通道数mid_nc，卷积核大小4x4，步长2，填充1
        # 公式：output_size = (input_size - kernel_size + 2*padding)/stride + 1
        self.conv2 = nn.Conv2d(
            mid_nc, mid_nc * 2, kernel_size=4, stride=2, padding=(1, 2)
        )
        # padding=(1,2)表示在高度方向填充1，宽度方向填充2
        # 这样设计是为了适应输入数据的宽高比例
        # mid_nc * 2表示输出通道数是mid_nc的两倍
        self.conv3 = nn.Conv2d(
            mid_nc * 2, mid_nc * 4, kernel_size=4, stride=2, padding=1
        )  # 标准层
        self.conv4 = nn.Conv2d(
            mid_nc * 4, mid_nc * 8, kernel_size=4, stride=1, padding=2
        )  # 不再下采样
        # 只增加感受野（网络中某一层的一个神经元，在输入图像上“能看到”的区域大小）
        self.conv5 = nn.Conv2d(mid_nc * 8, 1, kernel_size=4, stride=1, padding=1)
        # 最终感受野是70*70，意味着判别器输出特征图上的一个点实际在判断输入图像中一个70*70的patch是真还是假。最终输出的是一张真假热力图。
        # 这个尺寸蛮适合256*256的图像

        self.instance_norm1 = nn.InstanceNorm2d(
            mid_nc * 2, affine=True
        )  # 对每个样本的每个通道进行归一化处理
        self.instance_norm2 = nn.InstanceNorm2d(mid_nc * 4, affine=True)
        self.instance_norm3 = nn.InstanceNorm2d(mid_nc * 8, affine=True)

    def forward(self, x):
        h0 = F.leaky_relu(
            self.conv1(x), 0.2
        )  # 激活函数使用leaky relu，避免神经元死亡问题；卷积+下采样
        # 直接接原始图像
        h1 = F.leaky_relu(self.instance_norm1(self.conv2(h0)), 0.2)
        # 经过卷积+下采样conv2后，进行归一化处理instance_norm1，再经过leaky relu激活
        h2 = F.leaky_relu(self.instance_norm2(self.conv3(h1)), 0.2)
        # 同上,感受野扩大，空间分辨率降低H/4W/4（迫使网络做抽象，不再精确描述细节），通道数翻倍（特征维度增加，每个点可以表达更多组合关系）
        # 特征从纹理到结构
        h3 = F.leaky_relu(self.instance_norm3(self.conv4(h2)), 0.2)
        # stride=1,不再下采样，继续扩大感受野
        h4 = self.conv5(h3)
        # 无归一化和激活
        # 无激活函数是因为后续会用BCEWithLogitsLoss，它内部包含了sigmoid激活
        return h4
        # 输出是一张真假热力图，每个点表示对应patch的真假概率


# Residual Block for the Generator残差结构
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(dim, affine=True)
        self.instance_norm2 = nn.InstanceNorm2d(dim, affine=True)
        # instance normalization有助于风格迁移任务中的训练稳定性和收敛速度

    def forward(self, x):
        y = F.relu(self.instance_norm1(self.conv1(x)))
        y = self.instance_norm2(self.conv2(y))
        return x + y

    # 这里的x是输入，y是经过两层卷积和归一化后的输出，提供特征变化能力即修正
    # 残差连接：输入x直接加到输出y上，帮助梯度流动，缓解深层网络的训练难题。防止梯度消失，加快收敛，增加网络表达能力。
    # 只学变化即y，不学整个映射


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, mid_nc, output_nc):
        super(ResnetGenerator, self).__init__()
        self.conv1 = nn.Conv2d(
            input_nc, mid_nc, kernel_size=7, stride=1, padding=3
        )  # 浅层特征提取
        self.conv2 = nn.Conv2d(mid_nc, mid_nc * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            mid_nc * 2, mid_nc * 4, kernel_size=3, stride=2, padding=1
        )
        # 下采样两次，宽高减半两次，通道数翻倍两次
        self.instance_norm1 = nn.InstanceNorm2d(mid_nc, affine=True)
        self.instance_norm2 = nn.InstanceNorm2d(mid_nc * 2, affine=True)
        self.instance_norm3 = nn.InstanceNorm2d(mid_nc * 4, affine=True)

        # Define 9 residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(mid_nc * 4) for _ in range(9)]
        )
        # 9个残差块，每个块的输入输出通道数都是mid_nc * 4
        # 这样设计是因为经过两次下采样后，特征图已经足够抽象，适合进行复杂的特征变换--主干变换

        # 没有上下采样，通道数不变，保持空间尺寸不变，不负责结构变化，只负责特征变换
        # 在稳定结构的高语义特征空间中完成域间变换
        # 从代码结构可以看出，Residual Blocks 前后特征的空间尺寸与通道数完全一致，且通过恒等映射进行残差学习，因此该模块不会改变输入的几何结构，而是在高语义特征空间中完成输入域到输出域（目标域）的分布映射。

        self.deconv1 = nn.ConvTranspose2d(
            mid_nc * 4, mid_nc * 2, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            mid_nc * 2, mid_nc, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # 两次转置卷积用于上采样，宽高翻倍，通道数减半
        self.conv4 = nn.Conv2d(
            mid_nc, output_nc, kernel_size=7, stride=1, padding=(3, 2)
        )
        # 最后一个卷积层将通道数变回output_nc，padding=(3,2)适应输入数据宽高比例
        self.instance_norm4 = nn.InstanceNorm2d(mid_nc * 2, affine=True)
        self.instance_norm5 = nn.InstanceNorm2d(mid_nc, affine=True)

    def forward(self, x):
        c1 = F.relu(self.instance_norm1(self.conv1(x)))
        c2 = F.relu(self.instance_norm2(self.conv2(c1)))
        c3 = F.relu(self.instance_norm3(self.conv3(c2)))
        # c2c3编码 + 压缩 + 语义提升
        r = self.residual_blocks(c3)
        # → 主干变换 + 域映射输入域 → 输出域（不改结构）
        d1 = F.relu(self.instance_norm4(self.deconv1(r)))
        d2 = F.relu(self.instance_norm5(self.deconv2(d1)))
        pred = torch.tanh(self.conv4(d2))
        # → 解码 + 空间恢复
        # 输出生成结果。使用tanh激活函数将输出值限制在[-1, 1]范围内，适合图像数据表示
        return pred

    # 该生成器采用编码-残差-解码结构，通过两次下采样压缩空间维度，在残差块中完成主要域映射，再通过对称上采样恢复分辨率，既保证了全局结构一致性，又增强了局部细节表达能力。


class ImagePool:
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand() * self.maxsize)
            tmp1 = copy.copy(self.images[idx][0])
            self.images[idx][0] = image[0]
            idx = int(np.random.rand() * self.maxsize)
            tmp2 = copy.copy(self.images[idx][1])
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image
