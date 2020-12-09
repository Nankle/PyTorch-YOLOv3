# Learning Modulated Loss for Rotated Object Detection

论文链接🔗：

[Learning Modulated Loss for Rotated Object Detection.pdf](Learning%20Modulated%20Loss%20for%20Rotated%20Object%20Detecti%20ae0306c48ce644a1b01b5ad773d88859/Learning_Modulated_Loss_for_Rotated_Object_Detection.pdf)

## 1. 论文提出方法的切入点：

      Five parameters methods for Rotated Object Detection(coordinates of the central point, width, height, and rotation angle). **Traditional method**

      Aforementioned integration(前面提到的集成方法) can cause training instability and performance degeneration, due to the loss discontinuity resulted from the **inherent periodicity(固有周期性)** of angles and the associated sudden exchange of width and height.

解读：五参数法的旋转目标检测问题，在网络参数回归过程中，由于角度变量具有周期性，Bounding Box的宽、高进入网络的顺序随机，会导致Loss存在震荡问题发生，即无法控制Loss保持一种平稳的状态下降。

[本论文使用的主要数据集](Learning%20Modulated%20Loss%20for%20Rotated%20Object%20Detecti%20ae0306c48ce644a1b01b5ad773d88859/%E6%9C%AC%E8%AE%BA%E6%96%87%E4%BD%BF%E7%94%A8%E7%9A%84%E4%B8%BB%E8%A6%81%E6%95%B0%E6%8D%AE%E9%9B%86%20d6ad3cc4b3be485d9315039e93d7603a.csv)

---

## 2. RSE（Rotation Sensitive Error）

### 1. 什么是L1-Loss

        平均绝对误差（MAE）用于回归模型的损失函数，MAE是目标变量和预测变量之间绝对差值之和。因此它衡量的是一组预测值中的平均误差大小，而不考虑它们的方向（如果我们考虑方向的话，那就是均值误差（MBE）了，即误差之和），范围为0到∞。

![Learning%20Modulated%20Loss%20for%20Rotated%20Object%20Detecti%20ae0306c48ce644a1b01b5ad773d88859/Untitled.png](Learning%20Modulated%20Loss%20for%20Rotated%20Object%20Detecti%20ae0306c48ce644a1b01b5ad773d88859/Untitled.png)

### 2. RSE具体指代

i) The adoption of angle parameter and the resulting height-width exchange (in the popular five-parameter description in OpenCV) contribute to the sudden loss change (increase) in the boundary case.  **Loss Discontinuity**

ii) Regression inconsistency of measure units exists in the five-parameter model. 

    **Regression Inconsistency**

即这两方面：

1. 引入角度回归参量后，大家喜爱使用的Opencv中的定义矩形框角度的方式（x轴逆时针旋转遇到的第一条边，而不是相对于最长边）（-90，0），会导致长、宽的互换，造成Loss不稳定。
2. 角度参量与中心点坐标、width、height不同的计量单位集成在一个回归任务中会导致Loss不稳定，造成网络能力退化。

![Learning%20Modulated%20Loss%20for%20Rotated%20Object%20Detecti%20ae0306c48ce644a1b01b5ad773d88859/Untitled%201.png](Learning%20Modulated%20Loss%20for%20Rotated%20Object%20Detecti%20ae0306c48ce644a1b01b5ad773d88859/Untitled%201.png)