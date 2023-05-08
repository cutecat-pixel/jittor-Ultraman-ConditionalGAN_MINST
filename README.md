| 第三届计图挑战赛

# Jittor Conditional GAN on MINST

![主要结果](result.png)

## 简介

本项目包含了第三届计图挑战赛热身赛生成特定数字的图像比赛的代码实现。本项目的特点是：通过在生成器generator和判别器discriminator中添加相同的额外信息y，将GAN扩展为一个conditional模型，在MNIST数据集下进行训练从而生成特定的手写数字。

## 安装

#### 运行环境

- ubuntu 20.04 LTS
- python >= 3.7
- jittor >= 1.3.0

#### 安装依赖

执行以下命令安装 python 依赖

```
pip install jittor
```

参考Jittor官网：[Jittor(计图): 即时编译深度学习框架 — Jittor (tsinghua.edu.cn)](https://cg.cs.tsinghua.edu.cn/jittor/)

#### 预训练模型

本工程无预训练模型

## 数据预处理

```
from jittor.dataset.mnist import MNIST
import jittor.transform as transform

transform = transform.Compose([
    transform.Resize(opt.img_size),
    transform.Gray(),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),
])
dataloader = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=True)
```

## 训练

训练可运行以下命令：

```
python CGAN.py
```

## 推理

完成训练后会在根目录生成result.png

## 致谢

此项目基于官方提供的CGAN baseline实现，部分代码参考了 [JGAN/cgan.py at master · Jittor/JGAN · GitHub](https://github.com/Jittor/JGAN/blob/master/models/cgan/cgan.py)。

