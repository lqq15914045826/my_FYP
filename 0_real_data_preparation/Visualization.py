import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os


# 最简单的可视化 - 只看一眼
def quick_look():
    print("快速查看CSI数据集")

    # 加载数据
    data = scipy.io.loadmat("dataset_lab_150.mat")

    # 1. 基本信息
    print("1. 基本信息:")
    for key in data.keys():
        if not key.startswith("__"):
            print(f"   {key}: {data[key].shape}")

    # 2. 看一个csi样本
    csi1 = data["csi1"]
    print(f"\n2. csi1样本:")
    print(f"   形状: {csi1.shape}")
    print(f"   数据类型: {csi1.dtype}")

    # 取第一个样本
    if len(csi1.shape) == 4:
        # 取绝对值，将复数转换为实数
        csi1_abs = np.abs(csi1)
        sample = csi1_abs[:, :, :, 0]  # (时间, 子载波, 天线)
        print(f"   单个样本形状: {sample.shape}")
        print(f"   幅度范围: {sample.min():.4f} ~ {sample.max():.4f}")

        # 显示图片
        plt.figure(figsize=(15, 4))

        # 显示三个天线
        for i in range(3):
            plt.subplot(1, 4, i + 1)
            plt.imshow(sample[:, :, i].T, cmap="viridis", aspect="auto")
            plt.title(f"天线{i+1}")
            plt.xlabel("时间")
            plt.ylabel("子载波")

        # 显示平均
        plt.subplot(1, 4, 4)
        avg_sample = np.mean(sample, axis=2)  # 平均所有天线
        plt.imshow(avg_sample.T, cmap="viridis", aspect="auto")
        plt.title("平均所有天线")
        plt.xlabel("时间")
        plt.ylabel("子载波")

        plt.suptitle("CSI数据示例 (第一个样本，取绝对值)")
        plt.tight_layout()
        plt.show()

        # 3. 再看一下相位信息
        csi1_angle = np.angle(csi1)
        sample_angle = csi1_angle[:, :, :, 0]

        plt.figure(figsize=(15, 4))
        plt.suptitle("CSI数据相位信息 (第一个样本)")

        for i in range(3):
            plt.subplot(1, 4, i + 1)
            plt.imshow(sample_angle[:, :, i].T, cmap="hsv", aspect="auto")
            plt.title(f"天线{i+1}相位")
            plt.xlabel("时间")
            plt.ylabel("子载波")

        plt.subplot(1, 4, 4)
        avg_angle = np.mean(sample_angle, axis=2)
        plt.imshow(avg_angle.T, cmap="hsv", aspect="auto")
        plt.title("平均相位")
        plt.xlabel("时间")
        plt.ylabel("子载波")

        plt.tight_layout()
        plt.show()

    # 3. 看标签
    labels = data["label"]
    print(f"\n3. 标签:")
    print(f"   形状: {labels.shape}")
    print(f"   前10个标签: {labels[:10].flatten()}")

    # 简单统计
    unique_labels = np.unique(labels)
    print(f"   共有 {len(unique_labels)} 种不同标签")
    print(f"   标签值: {unique_labels}")

    # 显示标签分布
    plt.figure(figsize=(10, 5))
    counts = np.bincount(labels.flatten().astype(int))
    plt.bar(range(len(counts)), counts, alpha=0.7)
    plt.xlabel("标签值")
    plt.ylabel("样本数量")
    plt.title("标签分布")
    plt.grid(True, alpha=0.3)

    # 在每个柱子上显示数量
    for i, count in enumerate(counts):
        if count > 0:
            plt.text(i, count, str(count), ha="center", va="bottom")

    plt.show()


# 运行
if __name__ == "__main__":
    quick_look()
