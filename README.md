# 阈值提取器：使用opencv来对图像进行分割，快速筛选阈值
使用方法：
画面总览
<img width="821" alt="e9731e1450e25c0f6479e8e9e122ced" src="https://github.com/13559323996/-/assets/107629304/672ac2dc-be9a-44a0-88c8-f7e85fca5955">
可以下拉选择摄像头的编号，默认为电脑自带的摄像头0号，还可以下拉选择分割图像时的模式，包括灰度图、RGB、HSV、LAB。支持最小最大阈值分割，阈值分为3段，筛选更加精细。
可以实时对视频流进行分割，也可以按下stop，锁定当前帧图像进行分割，减少摄像头波动带来的干扰。
