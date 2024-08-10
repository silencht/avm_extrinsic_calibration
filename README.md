### [BeVis DataSet](https://shaoxuan92.github.io/BeVIS/)

#### 0. Download

根据链接下载BeVis数据集文件，并解压。

> [BeVIS Benchmark Dataset](https://pan.baidu.com/s/1GYCvroh8Bw5NHjxFQ_d0Wg)
>
> (pw: 2bn0)

#### 1. BeVis Folder Structure

```bash
BeVIS
    |
    |----CalibrationData（包含标定内参结果，和标定所需的棋盘格图像集）
    |       |----CalibResults（标定结果）
    |       |----extrinCalib-x & intrinCalib-x（标定数据集）
    |
    |----SLAM-easy-01 (Building DX) （地下停车场行驶数据集）
    |       |----GroundtruthData（真值）
    |       |----Imu（惯性传感器）
    |       |----Mynt（前视相机图像流）
    |       |----SurroundImages（四路鱼眼相机图像流）
    |       |    |----1609489072.460500000_Back.jpg
    |       |    |----1609489072.460500000_Front.jpg
    |       |    |----1609489072.460500000_Right.jpg
    |       |    |----1609489072.460500000_Left.jpg ……
    |       |     
    |       |----Timestamp（时间戳）
    |         
    |----SLAM-moderate/difficult-0x（同上）
    |
Note:该数据集四路鱼眼相机图像文件名称中的方向标识错误，真实关联如下（calibev.py程序已进行修改）：
Front代表前视相机，Right代表左视相机，Left代表后视相机，Back代表右视相机。
```

#### 2. Calibration & Bev

##### 2.1 Calibev Folder Structure

```bash
Bevis_calib
    |
    |----doc（BeVis数据集论文，数据格式等）
    |      |----BeVis.pdf
    |      |----BeVis_time_and_format.txt
    |      |----remap_table_test.py（BeVis标定结果文件夹中映射文件back_table.txt使用示例，效果差，原因未知）
    |      
    |----extrinsic（标定外参使用，保存外参H用文件夹）
    |       |----Back&Front&Left&Right（标定外参数据集，来源于Bevis--extrinCalib-x）
    |       |----undistor_Back/_Fr*/_Ri*/_Le*（程序中途生成，保存去畸变后的标定外参图片）
    |       |----SurroundImages（四路鱼眼相机图像流）
    |         
    |----intrinsic（保存内参K、畸变参数D文件用文件夹）
    |----picture 
    |       |----dst.jpg（BEV俯视棋盘格目标图，640*640）
    |       |----car.png（BEV中央填充汽车图）
    |----calibev.py（去畸、标定、拼接脚本程序）
    |----calibev.yaml（路径参数文件）
```

##### 2.2 Calibev

1. 选择相信BeVis数据集标定结果中的内参数据（内参数K&D已经嵌入calibev.py程序中）（./BeVIS/CalibrationData/CalibResults/CalibResults/surround-view-system/intrinsics.xml）

2. extrinsic中的图片来源于./Bevis/CalibrationData/extrinCalib-1/Surround-view System/surround-view system文件夹，根据图像命名的方向信息分类放入./avm_extrinsic_calibration/extrinsic/Back & Front & Left & Right四个文件夹。 其实，目前标定脚本程序中实际只使用了相同时间戳的一组图片（前后左右四张）

3. dst.jpg为640*640像素图，对应数据集车辆四周地面铺设的10米 \* 10米棋盘格标定布（每个格子长度为1米）（论文有描述）。该图片使用[draw.io](https://draw.io)软件绘制生成。

4. **在主目录下运行脚本程序**，例如：

   ```bash
   silencht@ubuntu:~/avm_extrinsic_calibration$ python3 calibev.py
   ```

   程序流程如下：

   1. 首先，程序检查./avm_extrinsic_calibration/extrinsic/Back & Front & Left & Right四个文件夹内图片之前是否去畸变，如果没有去畸变，那么执行去畸变，并将去畸变图片保存至./avm_extrinsic_calibration/extrinsic/undistor_Back/Fr\*/Le\*/Ri\*中。该过程中，同时会保存四路相机内参文件至./calib/intrinsic/（如front_D.npy，front_K.npy）（内参和畸变参数来源于BeVis数据集原始数据）。
   2. 其次，开始标定相机外参。暂只选取一组（前后左右共四张）图片进行手动标定来计算去畸变后相机原图与目标俯视图片（即./picture/dst.jpg）的单应性矩阵（矩阵包含外参R&t）。使用鼠标在去畸变原图上选取合适数量的棋盘格角点之后（大于等于4个），按空格跳出，然后使用鼠标在目标俯视图片再次点选对应匹配的角点，同样按空格结束。该步骤，前后左右四组图片共重复四次。【由于数据集的图片名称方向标识错误，因此在选取目标俯视图片的对应匹配角点时，需参照命令行中输出的真实方向（truth directory）进行点选。】最终完成后，会保存四路相机外参文件至./calib/extrinsic/（如back_H.npy）
   3. 最后，进行BEV拼接。读取calibev.yaml中bev_source_img_path下的四路鱼眼图片及汽车填充图./avm_extrinsic_calibration/picture/car.png，使用四路相机内外参（K、D、H）生成映射map矩阵。最终输出融合拼接过后的BEV俯视效果图至calibev.yaml中bev_img_path路径下。

5. 完整程序流程输入+输出示例如下

```bash
silencht@ubuntu:~/avm_extrinsic_calibration$ python3 calibev.py 
{'version': 0.1, 'extrinsic_path': './extrinsic/', 'intrinsic_path': './intrinsic/', 'dst_img_path': './picture/', 'bev_source_img_path': '/home/silencht/Videos/BeVis/SurroundImages/', 'bev_img_path': '/home/silencht/Videos/BeVis/BevImages/'}
mkdir undistor_img_save filefolder success!
bev_img_path filefolder is exists.
Undistortion Start...
[[421.50534803   0.         629.39810031]
 [  0.         419.99206255 531.4147271 ]
 [  0.           0.           1.        ]]
[-0.06658569 -0.00048144 -0.00119309  0.00016236]
[[419.61460581   0.         634.32006842]
 [  0.         418.50638109 539.32313432]
 [  0.           0.           1.        ]]
[-0.06699339 -0.00517398  0.00785958 -0.0042368 ]
[[420.86261222   0.         640.86939039]
 [  0.         419.49874064 535.82096052]
 [  0.           0.           1.        ]]
[-0.06544541 -0.00648174  0.00464294 -0.00147637]
[[423.15252271   0.         635.18368429]
 [  0.         421.7616208  546.04808802]
 [  0.           0.           1.        ]]
[-0.06850732  0.00331283 -0.00387445  0.00073377]
./extrinsic/Front/1609492906.591220000_Front.jpg
……
./extrinsic/Front/1609492904.201700000_Front.jpg
./extrinsic/Right/1609492904.633140000_Right.jpg
……
./extrinsic/Right/1609492905.97780000_Right.jpg
./extrinsic/Left/1609492905.529200000_Left.jpg
……
./extrinsic/Left/1609492899.555440000_Left.jpg
./extrinsic/Back/1609492899.854120000_Back.jpg
……
./extrinsic/Back/1609492904.633140000_Back.jpg
Undistortion Completed.
Extrinsic Calibrate Start...
./extrinsic/undistor_Front/1609492906.591220000_Front.jpg 
 --> truth_direct:  front
src_img points: 
 [[528. 505.]
 [710. 483.]
 [404. 571.]
 [781. 539.]]
dst_img points: 
 [[194.  64.]
 [381.  61.]
 [193. 191.]
 [385. 188.]]
homography: 
 [[-1.94026439e-01 -6.19648306e-01  3.97218214e+02]
 [-7.06295242e-02 -5.66598026e-01  3.17436717e+02]
 [-2.85973787e-04 -1.86646191e-03  1.00000000e+00]]
./extrinsic/undistor_Right/1609492904.633140000_Right.jpg 
 --> truth_direct:  left
src_img points: 
 [[544. 396.]
 [747. 430.]
 [439. 466.]
 [790. 526.]]
dst_img points: 
 [[ 65. 446.]
 [ 64. 255.]
 [190. 448.]
 [193. 257.]]
homography: 
 [[ 3.60500248e-01 -2.10088474e+00  5.90113713e+02]
 [ 9.85391638e-01 -1.80879731e+00 -1.33509787e+02]
 [ 1.04055500e-03 -5.73110072e-03  1.00000000e+00]]
./extrinsic/undistor_Left/1609492905.529200000_Left.jpg 
 --> truth_direct:  back
src_img points: 
 [[380. 512.]
 [778. 496.]
 [112. 640.]
 [928. 602.]]
dst_img points: 
 [[447. 577.]
 [255. 573.]
 [445. 511.]
 [258. 513.]]
homography: 
 [[ 6.90312815e-02 -7.47516822e-01  2.54260748e+02]
 [-6.50686536e-02 -1.04676488e+00  4.28700669e+02]
 [-1.36531824e-04 -2.29850340e-03  1.00000000e+00]]
./extrinsic/undistor_Back/1609492899.854120000_Back.jpg 
 --> truth_direct:  right
src_img points: 
 [[475. 476.]
 [675. 428.]
 [364. 594.]
 [743. 510.]]
dst_img points: 
 [[577. 192.]
 [576. 383.]
 [447. 193.]
 [447. 382.]]
homography: 
 [[-1.73642819e-01 -6.14728738e-01  2.60496879e+02]
 [-3.58466766e-01 -6.28249925e-01  4.31186766e+02]
 [-5.25526789e-04 -1.99365227e-03  1.00000000e+00]]
Extrinsic Calibrate Completed.
Remap to BEV Start...
There are  4033  images to be process : please wait...
100
200
……
4000
Remap Completed.
```

##### 2.3 decomposeH tools

decomposeH.py脚本程序可以将./avm_extrinsic_calibration/extrinsic/下的back_H.npy等外参文件转换分解为四组旋转和平移矩阵结果（详见《视觉SLAM十四讲》单应性矩阵相关内容）。输出结果如下所示例：

```bash
silencht@ubuntu:~/avm_extrinsic_calibration$ python3 decomposeH.py
[[ 9.86564375e-02 -7.89359033e-01  2.45322812e+02]
 [-3.97158701e-02 -1.08425352e+00  4.13379297e+02]
 [-9.27517179e-05 -2.41620887e-03  1.00000000e+00]]
[[423.15252271   0.         635.18368429]
 [  0.         421.7616208  546.04808802]
 [  0.           0.           1.        ]]
-----Rotation_0-------
[[-0.78676144 -0.22854702 -0.57338703]
 [ 0.03135011 -0.94252374  0.33266525]
 [-0.61646054  0.24375245  0.74870631]]
-----Translation_0-------
[[-7.20397629]
 [-1.36632877]
 [ 9.95481732]]
-----Rotation_1-------
[[-0.78676144 -0.22854702 -0.57338703]
 [ 0.03135011 -0.94252374  0.33266525]
 [-0.61646054  0.24375245  0.74870631]]
-----Translation_1-------
[[ 7.20397629]
 [ 1.36632877]
 [-9.95481732]]
-----Rotation_2-------
[[-0.89651757 -0.40081784  0.18868308]
 [ 0.11693784  0.19669711  0.97346586]
 [-0.4272959   0.89479344 -0.12947164]]
-----Translation_2-------
[[-7.29071086]
 [-2.65986019]
 [ 9.62461404]]
-----Rotation_3-------
[[-0.89651757 -0.40081784  0.18868308]
 [ 0.11693784  0.19669711  0.97346586]
 [-0.4272959   0.89479344 -0.12947164]]
-----Translation_3-------
[[ 7.29071086]
 [ 2.65986019]
 [-9.62461404]]
```

