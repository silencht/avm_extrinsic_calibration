# coding=utf-8
import os
import numpy as np
import cv2 as cv
import yaml
import argparse

YAML_FILE = "calibev.yaml"

#--------------------------------------------------------------参数初始化--------------------------------------------------------------------------------------------
bev_image_count = 1

parser = argparse.ArgumentParser(description="Camera Intrinsic Calibration")
parser.add_argument('-image', '--IMAGE_FILE', default='1609', type=str, help='Input Image File Name Prefix (eg.: img_raw)')
parser.add_argument('-fw','--FRAME_WIDTH', default=1280, type=int, help='Camera Frame Width')
parser.add_argument('-fh','--FRAME_HEIGHT', default=1080, type=int, help='Camera Frame Height')
parser.add_argument('-bw', '--BEV_WIDTH', default=640, type=int, help='BEV Frame Width')
parser.add_argument('-bh', '--BEV_HEIGHT', default=640, type=int, help='BEV Frame Height')
parser.add_argument('-cw', '--CAR_WIDTH', default=150, type=int, help='Car Frame Width')                                          # 鸟瞰图中间车辆宽度（与car图片相对应）
parser.add_argument('-ch', '--CAR_HEIGHT', default=330, type=int, help='Car Frame Height')                                        # 鸟瞰图中间车辆高度（与car图片相对应）
parser.add_argument('-fs', '--FOCAL_SCALE', default=0.5, type=float, help='Camera Undistort Focal Scale')
parser.add_argument('-ss', '--SIZE_SCALE', default=1, type=float, help='Camera Undistort Size Scale')
parser.add_argument('-subpix_s','--SUBPIX_REGION_SRC', default=3, type=int, help='Corners Subpix Region of img_src')
parser.add_argument('-subpix_d','--SUBPIX_REGION_DST', default=3, type=int, help='Corners Subpix Region of img_dst')
parser.add_argument('-blend','--BLEND_FLAG', default=True, type=bool, help='Blend BEV Image (Ture/False)')                        # 鸟瞰图拼接是否采用图像融合
parser.add_argument('-balance','--BALANCE_FLAG', default=True, type=bool, help='Balance BEV Image (Ture/False)')                  # 鸟瞰图拼接是否采用图像平衡
args = parser.parse_args([])              # Jupyter Notebook中直接运行时要加[], py文件则去掉


# 直接赋值避免频繁读取args
FRAME_WIDTH = args.FRAME_WIDTH
FRAME_HEIGHT = args.FRAME_HEIGHT
BEV_WIDTH = args.BEV_WIDTH
BEV_HEIGHT = args.BEV_HEIGHT
CAR_WIDTH = args.CAR_WIDTH
CAR_HEIGHT = args.CAR_HEIGHT
FOCAL_SCALE = args.FOCAL_SCALE
SIZE_SCALE = args.SIZE_SCALE

# 从config yaml文件读取路径参数
with open(YAML_FILE, 'r') as f:
    data = yaml.safe_load(f)
    print(data)
extrinsic_path = data["extrinsic_path"]
intrinsic_path = data["intrinsic_path"]
dst_img_path   = data["dst_img_path"]
bev_source_img_path = data["bev_source_img_path"]
bev_img_path   = data["bev_img_path"]

# 需要进行去畸变的图片路径
img_todistor_front_path_ = extrinsic_path+"Front/"
img_todistor_right_path_ = extrinsic_path+"Right/"
img_todistor_left_path_  = extrinsic_path+"Left/"
img_todistor_rear_path_  = extrinsic_path+"Back/"
img_todistor_paths_      = [img_todistor_front_path_,img_todistor_right_path_,img_todistor_left_path_,img_todistor_rear_path_]

# 去畸变后保存图片路径
img_distored_front_path_ = extrinsic_path+"undistor_Front/"
img_distored_right_path_ = extrinsic_path+"undistor_Right/"
img_distored_left_path_  = extrinsic_path+"undistor_Left/"
img_distored_back_path_  = extrinsic_path+"undistor_Back/"

# 经测试，BeVis中的前后左右实际对应是 前实际是（前/黑书包），后实际是（右/浅蓝色片），左实际是（后/粉红色扫把），右实际是（左/深蓝色包）
img_distored_paths_      = [img_distored_front_path_,img_distored_right_path_,img_distored_left_path_,img_distored_back_path_]
img_truth_directorys_    = ["front", "left", "back", "right"]
camera_truth_ids         = ["front", "left", "back", "right"]
camera_ids               = ["front","right","left","back"]

# 新建文件夹用以保存去畸变图片结果
isUndisExists=os.path.exists(img_distored_front_path_)
if not isUndisExists:
    for path_ in img_distored_paths_:
        os.makedirs(path_)
    print("mkdir undistor_img_save filefolder success!")
else:
    print("undistor_img_save filefolder is exists.")

# 外参文件是否存在
isExtrinExists = os.path.exists(extrinsic_path+'/back_H.npy')
if isExtrinExists:
    print("extrinsic file is exists.")

# 新建一个文件夹保存四合一图片结果
isBevExists=os.path.exists(bev_img_path)
if not isBevExists:
    os.makedirs(bev_img_path) 
    print("mkdir bev_img_path filefolder success!")
else:
    print("bev_img_path filefolder is exists.")


# 获取图片文件
def get_images(PATH, NAME):
    filePath = [os.path.join(PATH, x) for x in os.listdir(PATH) 
                if any(x.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
               ]                                                            # 得到给定路径下所有图片文件
    filenames = [filename for filename in filePath if NAME in filename]     # 再筛选出包含给定名字的图片
    if len(filenames) == 0:
        raise Exception("from {} read images failed".format(PATH))
    return filenames

#--------------------------------------------------------------图像去畸变------------------------------------------------------------------------------------------
# 标定数据类
class CalibData:                             
    def __init__(self):
        self.type = None                     # 自定义数据类型
        self.camera_intrinsic_mat = None     # 相机内参
        self.camera_extrinsic_mat = None     # 相机外参
        self.dist_coeff = None               # 畸变参数
        self.map1 = None                     # 映射矩阵1
        self.map2 = None                     # 映射矩阵2
        self.ok = False                      # 数据采集完成标志

# 四路鱼眼相机数据
class Four_Fisheye:                               
    def __init__(self):
        self.f_data = CalibData()            # front camera
        self.r_data = CalibData()            # right camera
        self.l_data = CalibData()            # left  camera
        self.b_data = CalibData()            # back  camera

    # 初始化内参
    def intrinsic_mat_init(self):

        f_data = self.f_data
        f_data.type = "FISHEYE"
        f_data.camera_intrinsic_mat = np.array([[4.2150534803053478e+02,0.,6.2939810031193633e+02],
                                      [0.,4.1999206255343978e+02,5.3141472710260518e+02],
                                      [0.,0.,1.]]) # front_camera intrinsic
        f_data.dist_coeff = np.array([-6.6585685927759056e-02,-4.8144285824610098e-04,-1.1930897697190990e-03,1.6236147741932646e-04]).T # front_cameara undistortion
        f_data.ok = cv.checkRange(f_data.camera_intrinsic_mat) and cv.checkRange(f_data.dist_coeff)
        self._get_undistort_maps("front")
        self._save_intrinsic_mat("front",f_data.camera_intrinsic_mat,f_data.dist_coeff)

        r_data = self.r_data      
        r_data.type = "FISHEYE"
        r_data.camera_intrinsic_mat = np.array([[4.1961460580570463e+02,0.,6.3432006841655129e+02],
                                      [0.,4.1850638109014426e+02,5.3932313431747673e+02],
                                      [0.,0.,1.]]) # right_camera intrinsic
        r_data.dist_coeff = np.array([-6.6993385910155065e-02,-5.1739781929103605e-03,7.8595773802962888e-03,-4.2367990313813440e-03]).T # right_cameara undistortion
        r_data.ok = cv.checkRange(r_data.camera_intrinsic_mat) and cv.checkRange(r_data.dist_coeff)
        self._get_undistort_maps("right")
        self._save_intrinsic_mat("right",r_data.camera_intrinsic_mat,r_data.dist_coeff)

        l_data = self.l_data       
        l_data.type = "FISHEYE"
        l_data.camera_intrinsic_mat = np.array([[4.2086261221668570e+02,0.,6.4086939039393337e+02],
                                      [0.,4.1949874063802940e+02,5.3582096051915732e+02],
                                      [0.,0.,1.]]) # left_camera intrinsic
        l_data.dist_coeff = np.array([-6.5445414949742764e-02,-6.4817440226779821e-03,4.6429370436962608e-03,-1.4763681169119418e-03]).T # left_cameara undistortion
        l_data.ok = cv.checkRange(l_data.camera_intrinsic_mat) and cv.checkRange(l_data.dist_coeff)
        self._get_undistort_maps("left")
        self._save_intrinsic_mat("left",l_data.camera_intrinsic_mat,l_data.dist_coeff)

        b_data = self.b_data       
        b_data.type = "FISHEYE"
        b_data.camera_intrinsic_mat = np.array([[4.2315252270666946e+02,0.,6.3518368429424913e+02],
                                      [0.,4.2176162080058998e+02,5.4604808802459536e+02],
                                      [0.,0.,1.]]) # back_camera intrinsic
        b_data.dist_coeff = np.array([-6.8507324567971206e-02,3.3128278165505034e-03,-3.8744468086803550e-03,7.3376684970524460e-04]).T # back_cameara undistortion
        b_data.ok = cv.checkRange(b_data.camera_intrinsic_mat) and cv.checkRange(b_data.dist_coeff)
        self._get_undistort_maps("back")
        self._save_intrinsic_mat("back",b_data.camera_intrinsic_mat,b_data.dist_coeff)

    # 初始化外参（外参标定之后才可使用）
    # def extrinsic_mat_init(self):
    #     f_data = self.f_data
    #     r_data = self.r_data
    #     l_data = self.l_data
    #     b_data = self.b_data

    #     f_data.camera_extrinsic_mat = np.load('front_H.npy')
    #     r_data.camera_extrinsic_mat = np.load('right_H.npy')
    #     l_data.camera_extrinsic_mat = np.load('left_H.npy')
    #     b_data.camera_extrinsic_mat = np.load('back_H.npy')        

    # 计算去畸变的新的相机内参，可以改变焦距和画幅
    def _get_camera_mat_dst(self, camera_intrinsic_mat):
        camera_mat_dst = camera_intrinsic_mat.copy()
        camera_mat_dst[0][0] *= FOCAL_SCALE
        camera_mat_dst[1][1] *= FOCAL_SCALE
        camera_mat_dst[0][2] = FRAME_WIDTH  / 2 * SIZE_SCALE
        camera_mat_dst[1][2] = FRAME_HEIGHT / 2 * SIZE_SCALE
        return camera_mat_dst

    # 计算去畸变的映射矩阵
    def _get_undistort_maps(self,camera_id):
        if  camera_id == "front":
            f_data = self.f_data
            camera_mat_dst = self._get_camera_mat_dst(f_data.camera_intrinsic_mat)
            print(f_data.camera_intrinsic_mat)
            print(f_data.dist_coeff)
            f_data.map1, f_data.map2 = cv.fisheye.initUndistortRectifyMap(
                                        f_data.camera_intrinsic_mat, f_data.dist_coeff, np.eye(3, 3), camera_mat_dst, 
                                        (int(FRAME_WIDTH * SIZE_SCALE), int(FRAME_HEIGHT * SIZE_SCALE)), cv.CV_16SC2)
        elif camera_id == "right":
            r_data = self.r_data
            camera_mat_dst = self._get_camera_mat_dst(r_data.camera_intrinsic_mat)
            print(r_data.camera_intrinsic_mat)
            print(r_data.dist_coeff)
            r_data.map1, r_data.map2 = cv.fisheye.initUndistortRectifyMap(
                                        r_data.camera_intrinsic_mat, r_data.dist_coeff, np.eye(3, 3), camera_mat_dst, 
                                        (int(FRAME_WIDTH * SIZE_SCALE), int(FRAME_HEIGHT * SIZE_SCALE)), cv.CV_16SC2)                                        
        elif camera_id == "left":
            l_data = self.l_data
            camera_mat_dst = self._get_camera_mat_dst(l_data.camera_intrinsic_mat)
            print(l_data.camera_intrinsic_mat)
            print(l_data.dist_coeff)            
            l_data.map1, l_data.map2 = cv.fisheye.initUndistortRectifyMap(
                                        l_data.camera_intrinsic_mat, l_data.dist_coeff, np.eye(3, 3), camera_mat_dst, 
                                        (int(FRAME_WIDTH * SIZE_SCALE), int(FRAME_HEIGHT * SIZE_SCALE)), cv.CV_16SC2)
        elif camera_id == "back":
            b_data = self.b_data
            camera_mat_dst = self._get_camera_mat_dst(b_data.camera_intrinsic_mat)
            print(b_data.camera_intrinsic_mat)
            print(b_data.dist_coeff)            
            b_data.map1, b_data.map2 = cv.fisheye.initUndistortRectifyMap(
                                        b_data.camera_intrinsic_mat, b_data.dist_coeff, np.eye(3, 3), camera_mat_dst, 
                                        (int(FRAME_WIDTH * SIZE_SCALE), int(FRAME_HEIGHT * SIZE_SCALE)), cv.CV_16SC2)   

    def _save_intrinsic_mat(self,camera_id,camera_intrinsic_mat,dist_coeff):
        np.save(intrinsic_path+camera_id+"_K.npy",camera_intrinsic_mat.tolist())
        np.save(intrinsic_path+camera_id+"_D.npy",dist_coeff.tolist())

#去畸变器
class Undistortor:
    def __init__(self):
        self.four_camera = Four_Fisheye()
        self.four_camera.intrinsic_mat_init()

    def undistort(self, img, camera_id):
        if   camera_id == "front":
            data = self.four_camera.f_data
            return cv.remap(img, data.map1, data.map2, cv.INTER_LINEAR)
        elif camera_id == "right":
            data = self.four_camera.r_data
            return cv.remap(img, data.map1, data.map2, cv.INTER_LINEAR)
        elif camera_id == "left":
            data = self.four_camera.l_data
            return cv.remap(img, data.map1, data.map2, cv.INTER_LINEAR)                       
        elif camera_id == "back":
            data = self.four_camera.b_data
            return cv.remap(img, data.map1, data.map2, cv.INTER_LINEAR)

    def __call__(self):
        for todistor_path_,tosave_path_,camera_id in zip(img_todistor_paths_,img_distored_paths_,camera_truth_ids):
            filenames = get_images(todistor_path_, args.IMAGE_FILE)            # 获取图片
            for filename in filenames:                                         # 遍历图片
                print(filename)
                raw_frame = cv.imread(filename)
                cv.namedWindow(camera_id, flags = cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
                cv.imshow(camera_id, raw_frame)
                cv.waitKey(1)
                undist_frame = self.undistort(raw_frame,camera_id)
                cv.imwrite(tosave_path_ + os.path.basename(filename), undist_frame)
                cv.namedWindow(camera_id+"-undist_frame", flags = cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
                cv.imshow(camera_id+"-undist_frame", undist_frame)   
                cv.waitKey(1)
            cv.destroyAllWindows() 
       

#------------------------------------------------------------开始标定相机外参---------------------------------------------------------------------------------------
# 鼠标取点
def on_EVENT_LBUTTONDOWN(event, x, y, flags, data):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(data['img'], (x, y), 1, (0, 0, 255), thickness=-1)
        cv.imshow("seclect your points", data['img'])
        data['img_points'].append([x, y])

# 外参标定器，从两张图像得到单应性矩阵
class ExCalibrator():           
    def __init__(self):
        self.src_corners = np.empty([0,1,2])
        self.dst_corners = np.empty([0,1,2])

    def get_src_img_corner_points(self, img):
        data = {}
        data['img_points'] = []
        data['img'] = img.copy()
        cv.namedWindow("seclect your points")
        cv.imshow("seclect your points", img)
        cv.setMouseCallback("seclect your points", on_EVENT_LBUTTONDOWN, data)
        cv.waitKey(0)
        points = np.vstack(data['img_points']).astype(float)
        print("src_img points: \n",points)
        cv.destroyAllWindows()
        return points

    def get_dst_img_corner_points(self, img):
        data = {}
        data['img_points'] = []
        data['img'] = img.copy()
        cv.namedWindow("seclect your points")
        cv.imshow("seclect your points", img)
        cv.setMouseCallback("seclect your points", on_EVENT_LBUTTONDOWN, data)
        cv.waitKey(0)
        points = np.vstack(data['img_points']).astype(float)
        print("dst_img points: \n",points)
        cv.destroyAllWindows()
        return points    

    def warp(self):
        src_warp = cv.warpPerspective(self.src_img, self.homography, (self.dst_img.shape[1], self.dst_img.shape[0])) 
        return src_warp
        
    def get_homography_mat(self, src_img, dst_img):
        self.src_img = src_img
        self.dst_img = dst_img
        self.src_corners = self.get_src_img_corner_points(self.src_img)
        self.dst_corners = self.get_dst_img_corner_points(self.dst_img)
        self.homography, mask = cv.findHomography(self.src_corners, self.dst_corners, method = cv.RANSAC)
        return self.homography

    def __call__(self):
        for path_, truth_direct_ in zip(img_distored_paths_, img_truth_directorys_):
            srcfiles = get_images(path_, args.IMAGE_FILE)
            dstfiles = get_images(dst_img_path, "dst")
            print(srcfiles[0],"\n --> truth_direct: ",truth_direct_)
            src_img = cv.imread(srcfiles[0]) # 此行的 0可换为图片数量范围内的其他整数
            dst_img = cv.imread(dstfiles[0])
            
            # 输入对应两张去畸变图像得到单应性矩阵
            homography = self.get_homography_mat(src_img, dst_img)
            print("homography: \n",homography)

            # 保存单应性矩阵
            homography_file_name = truth_direct_ + "_H.npy"
            np.save(extrinsic_path + homography_file_name, homography)

            # 得到source图像的变换结果图
            src_warp = self.warp()
            cv.namedWindow("Warped Source View", flags = cv.WINDOW_NORMAL|cv.WINDOW_KEEPRATIO)
            cv.imshow("Warped Source View", src_warp)
            cv.waitKey(0)
            cv.destroyAllWindows()    

#-------------------------------------------------------------todo:拼接 BEV---------------------------------------------------------------------------------------
# 图像补充黑边
def padding(img,width,height):
    H = img.shape[0]
    W = img.shape[1]
    top = (height - H) // 2        # //为向下取整除法
    bottom = (height - H) // 2 
    if top + bottom + H < height:
        bottom += 1
    left = (width - W) // 2 
    right = (width - W) // 2 
    if left + right + W < width:
        right += 1
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value = (0,0,0)) 
    return img

# 亮度平衡 根据输入的四张图像进行计算平均亮度 转化为HSV通道
def luminance_balance(images):
    [front,back,left,right] = [cv.cvtColor(image,cv.COLOR_BGR2HSV) 
                               for image in images]
    hf, sf, vf = cv.split(front)
    hb, sb, vb = cv.split(back)
    hl, sl, vl = cv.split(left)
    hr, sr, vr = cv.split(right)
    V_f = np.mean(vf)
    V_b = np.mean(vb)
    V_l = np.mean(vl)
    V_r = np.mean(vr)
    V_mean = (V_f + V_b + V_l +V_r) / 4
    vf = cv.add(vf,(V_mean - V_f))
    vb = cv.add(vb,(V_mean - V_b))
    vl = cv.add(vl,(V_mean - V_l))
    vr = cv.add(vr,(V_mean - V_r))
    front = cv.merge([hf,sf,vf])
    back = cv.merge([hb,sb,vb])
    left = cv.merge([hl,sl,vl])
    right = cv.merge([hr,sr,vr])
    images = [front,back,left,right]
    images = [cv.cvtColor(image,cv.COLOR_HSV2BGR) for image in images]
    return images

# 色彩平衡（白平衡）
def color_balance(image):
    b, g, r = cv.split(image)
    B = np.mean(b)
    G = np.mean(g)
    R = np.mean(r)
    K = (R + G + B) / 3
    Kb = K / B
    Kg = K / G
    Kr = K / R
    cv.addWeighted(b, Kb, 0, 0, 0, b)
    cv.addWeighted(g, Kg, 0, 0, 0, g)
    cv.addWeighted(r, Kr, 0, 0, 0, r)
    return cv.merge([b,g,r])

# 融合拼接四个图像的mask
class BlendMask:                
    def __init__(self,name):
        mf = self.get_mask('front')
        mb = self.get_mask('back')
        ml = self.get_mask('left')
        mr = self.get_mask('right')
        self.get_lines()
        if name == 'front':
            mf = self.get_blend_mask(mf, ml, self.lineFL, self.lineLF)
            mf = self.get_blend_mask(mf, mr, self.lineFR, self.lineRF)
            self.mask = mf
        if name == 'back':
            mb = self.get_blend_mask(mb, ml, self.lineBL, self.lineLB)
            mb = self.get_blend_mask(mb, mr, self.lineBR, self.lineRB)
            self.mask = mb
        if name == 'left':
            ml = self.get_blend_mask(ml, mf, self.lineLF, self.lineFL)
            ml = self.get_blend_mask(ml, mb, self.lineLB, self.lineBL)
            self.mask = ml
        if name == 'right':
            mr = self.get_blend_mask(mr, mf, self.lineRF, self.lineFR)
            mr = self.get_blend_mask(mr, mb, self.lineRB, self.lineBR)
            self.mask = mr
        self.weight = np.repeat(self.mask[:, :, np.newaxis], 3, axis=2) / 255.0
        self.weight = self.weight.astype(np.float32)
    
    # 预设好的前后左右四个mask的坐标点，相比直接拼接的mask，有重叠部分，修改数值可以改变重叠范围
    def get_points(self, name):
        if name == 'front':
            points = np.array([
                [0, 0],
                [BEV_WIDTH, 0], 
                [BEV_WIDTH, BEV_HEIGHT/5], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [0, BEV_HEIGHT/5], 
            ]).astype(np.int32)
        elif name == 'back':
            points = np.array([
                [0, BEV_HEIGHT],
                [BEV_WIDTH, BEV_HEIGHT],
                [BEV_WIDTH, BEV_HEIGHT - BEV_HEIGHT/5],
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2], 
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                [0, BEV_HEIGHT - BEV_HEIGHT/5],
            ]).astype(np.int32)
        elif name == 'left':
            points = np.array([
                [0, 0],
                [0, BEV_HEIGHT], 
                [BEV_WIDTH/5, BEV_HEIGHT], 
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [BEV_WIDTH/5, 0]
            ]).astype(np.int32)
        elif name == 'right':
            points = np.array([
                [BEV_WIDTH, 0],
                [BEV_WIDTH, BEV_HEIGHT], 
                [BEV_WIDTH - BEV_WIDTH/5, BEV_HEIGHT],
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [BEV_WIDTH - BEV_WIDTH/5, 0]
            ]).astype(np.int32)
        else:
            raise Exception("name should be front/back/left/right")
        return points
    
    # 填充得到各个mask
    def get_mask(self, name):
        mask = np.zeros((BEV_HEIGHT,BEV_WIDTH), dtype=np.uint8)
        points = self.get_points(name)
        return cv.fillPoly(mask, [points], 255)
    
    # 得到预设的mask重叠部分的各个线段
    def get_lines(self):
        self.lineFL = np.array([
                        [0, BEV_HEIGHT/5], 
                        [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                    ]).astype(np.int32)
        self.lineFR = np.array([
                        [BEV_WIDTH, BEV_HEIGHT/5], 
                        [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                    ]).astype(np.int32)
        self.lineBL = np.array([
                        [0, BEV_HEIGHT - BEV_HEIGHT/5], 
                        [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                    ]).astype(np.int32)
        self.lineBR = np.array([
                        [BEV_WIDTH, BEV_HEIGHT - BEV_HEIGHT/5], 
                        [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                    ]).astype(np.int32)
        self.lineLF = np.array([
                        [BEV_WIDTH/5, 0],
                        [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
                    ]).astype(np.int32)
        self.lineLB = np.array([
                        [BEV_WIDTH/5, BEV_HEIGHT],
                        [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2]
                    ]).astype(np.int32)
        self.lineRF = np.array([
                        [BEV_WIDTH - BEV_WIDTH/5, 0],
                        [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
                    ]).astype(np.int32)
        self.lineRB = np.array([
                        [BEV_WIDTH - BEV_WIDTH/5, BEV_HEIGHT],
                        [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2]
                    ]).astype(np.int32)
        
    # 根据重叠部分的点到上述线段的距离，得到该处的权重值
    def get_blend_mask(self, maskA, maskB, lineA, lineB):
        overlap = cv.bitwise_and(maskA, maskB)           # 重叠区域
        indices = np.where(overlap != 0)                  # 重叠区域的坐标索引
        for y, x in zip(*indices):
            distA = cv.pointPolygonTest(np.array(lineA), (int(x), int(y)), True)     # 到重叠区域边缘的距离A
            distB = cv.pointPolygonTest(np.array(lineB), (int(x), int(y)), True)     # 到重叠区域边缘的距离B
            maskA[y, x] = distA**2 / (distA**2 + distB**2 + 1e-6) * 255     # 根据距离的平方比值确定该处权重
        return maskA
    
    # 将图像乘以权重mask
    def __call__(self, img):
        return (img * self.weight).astype(np.uint8)    

# 直接拼接四个图像的mask
class Mask:                     
    def __init__(self, name):
        self.mask = self.get_mask(name)
    
    # 预设好的前后左右四个mask的坐标点
    def get_points(self, name):
        if name == 'front':
            points = np.array([
                [0, 0],
                [BEV_WIDTH, 0], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
            ]).astype(np.int32)
        elif name == 'back':
            points = np.array([
                [0, BEV_HEIGHT],
                [BEV_WIDTH, BEV_HEIGHT],
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2], 
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2]
            ]).astype(np.int32)
        elif name == 'left':
            points = np.array([
                [0, 0],
                [0, BEV_HEIGHT], 
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
            ]).astype(np.int32)
        elif name == 'right':
            points = np.array([
                [BEV_WIDTH, 0],
                [BEV_WIDTH, BEV_HEIGHT], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
            ]).astype(np.int32)
        else:
            raise Exception("name should be front/back/left/right")
        return points
    
    # 填充得到各个mask
    def get_mask(self, name):
        mask = np.zeros((BEV_HEIGHT,BEV_WIDTH), dtype=np.uint8)
        points = self.get_points(name)
        return cv.fillPoly(mask, [points], 255)
    
    # 位与计算得到mask后的图像
    def __call__(self, img):
        return cv.bitwise_and(img, img, mask=self.mask)

# 相机类 读取参数、去畸变和单应性变换
class Camera:                   
    def __init__(self, name):
        # 读取内参、畸变向量、外参的npy文件
        self.camera_intrinsic_mat = np.load('./intrinsic/{}_K.npy'.format(name))
        self.dist_coeff = np.load('./intrinsic/{}_D.npy'.format(name))
        self.camera_extrinsic_mat = np.load('./extrinsic/{}_H.npy'.format(name))
        self.camera_mat_dst = self.get_camera_mat_dst()
        self.undistort_maps = self.get_undistort_maps()
        self.bev_maps = self.get_bev_maps()
        # print(name,"\nK:\n",self.camera_intrinsic_mat,"\nD:\n",self.dist_coeff,"\nH:\n",self.camera_extrinsic_mat)
    
    # 获取去畸变的新的相机内参矩阵 可以修改焦距和画幅
    def get_camera_mat_dst(self):
        camera_mat_dst = self.camera_intrinsic_mat.copy()
        camera_mat_dst[0][0] *= FOCAL_SCALE
        camera_mat_dst[1][1] *= FOCAL_SCALE
        camera_mat_dst[0][2] = FRAME_WIDTH  / 2 * SIZE_SCALE
        camera_mat_dst[1][2] = FRAME_HEIGHT / 2 * SIZE_SCALE
        return camera_mat_dst

    # 获取去畸变映射矩阵
    def get_undistort_maps(self):
        undistort_maps = cv.fisheye.initUndistortRectifyMap(
                        self.camera_intrinsic_mat, self.dist_coeff, 
                        np.eye(3, 3), self.camera_mat_dst,
                        (int(FRAME_WIDTH * SIZE_SCALE), int(FRAME_HEIGHT * SIZE_SCALE)), cv.CV_16SC2)
        return undistort_maps

    # 对去畸变映射矩阵进行单应性变换
    def get_bev_maps(self):
        map1 = self.warp_homography(self.undistort_maps[0])
        map2 = self.warp_homography(self.undistort_maps[1])
        return (map1, map2)
    
    # 原始图像去畸变
    def undistort(self, img):
        return cv.remap(img, *self.undistort_maps, interpolation = cv.INTER_LINEAR)
        
    # 直接对图像进行单应性变换    
    def warp_homography(self, img):
        return cv.warpPerspective(img, self.camera_extrinsic_mat, (BEV_WIDTH,BEV_HEIGHT))
    
    # 利用单应性变换后的映射矩阵进行remap变换，速度更快
    def raw2bev(self, img):
        return cv.remap(img, *self.bev_maps, interpolation = cv.INTER_LINEAR)

# 环视鸟瞰图生成器
class BevGenerator:                   
    def __init__(self, blend=args.BLEND_FLAG, balance=args.BALANCE_FLAG):
        self.cameras = [Camera('front'), Camera('back'), 
                        Camera('left'),  Camera('right')]
        self.blend   = blend        # 鸟瞰图拼接是否采用图像融合
        self.balance = balance      # 鸟瞰图拼接是否采用图像平衡
        if not self.blend:
            self.masks = [Mask('front'), Mask('back'), 
                          Mask('left'),  Mask('right')]
        else:
            self.masks = [BlendMask('front'), BlendMask('back'), 
                          BlendMask('left'),  BlendMask('right')]
    
    # 输入前后左右四张原始相机图像，生成鸟瞰图，car图像可以不输入
    def __call__(self, front, back, left, right, car = None):
        images = [front, left, right, back]           # 方向bug重排
        if self.balance:
            images = luminance_balance(images)        # 亮度平衡
        images = [mask(camera.raw2bev(img)) 
                  for img, mask, camera in zip(images, self.masks, self.cameras)]   # 鸟瞰变换并加上mask
        surround = cv.add(images[0],images[1])
        surround = cv.add(surround,images[2])
        surround = cv.add(surround,images[3])        # 将所有图像拼接起来
        if self.balance:
            surround = color_balance(surround)        # 白平衡
        if car is not None:
            surround = cv.add(surround,car)          # 加上车辆图片
        return surround   

#------------------------------------------------------------Main()-----------------------------------------------------------------------------------------------
def main():
    # 1.如果没有去畸变，开始图片去畸变（通过去畸变图片保存路径的文件夹是否存在判断）
    if not isUndisExists:
        print("Undistortion Start...")
        undistortor = Undistortor()                                     # 初始化去畸变器
        undistortor()                                                   # 开始去畸变      
        print("Undistortion Completed.")
    else:
        print("Do not need to Undistortion.")

    # 2.开始标定相机外参
    if not isExtrinExists:
        print("Extrinsic Calibrate Start...")
        exCalibrator = ExCalibrator()                                   # 初始化外参标定器
        exCalibrator()                                                  # 开始计算外参
        print("Extrinsic Calibrate Completed.")
    else:
        print("Do not need to ExCalibrate.")

    # 3.拼接BEV
    if not isBevExists:
        print("Remap to BEV Start...")
        f_command = "ls"+" "+bev_source_img_path+" | grep Front"+" > "+bev_source_img_path+"image_front.txt" 
        b_command = "ls"+" "+bev_source_img_path+" | grep Back"+" > "+bev_source_img_path+"image_back.txt"
        l_command = "ls"+" "+bev_source_img_path+" | grep Left"+" > "+bev_source_img_path+"image_left.txt"
        r_command = "ls"+" "+bev_source_img_path+" | grep Right"+" > "+bev_source_img_path+"image_right.txt"
        os.system(f_command)
        os.system(b_command)
        os.system(l_command)
        os.system(r_command)

        # 从对应的.txt中读取图片名称
        image_front_path_ = os.path.join(bev_source_img_path,'image_front.txt')
        image_right_path_ = os.path.join(bev_source_img_path,'image_right.txt')
        image_left_path_  = os.path.join(bev_source_img_path,'image_left.txt')
        image_back_path_  = os.path.join(bev_source_img_path,'image_back.txt')
        with open(image_front_path_, 'r') as f:
            front_names = f.readlines()
        with open(image_right_path_, 'r') as f:
            right_names = f.readlines()
        with open(image_left_path_, 'r') as f:
            left_names  = f.readlines()     
        with open(image_back_path_, 'r') as f:
            back_names  = f.readlines()

        # 处理BEV中央填充车辆图片
        car_img = cv.imread('./picture/car.png')                            
        car_img = cv.resize(car_img,(CAR_WIDTH,CAR_HEIGHT),interpolation = cv.INTER_AREA)
        car_img = padding(car_img, BEV_WIDTH, BEV_HEIGHT)                   # 将车辆图片补充黑边至鸟瞰图大小

        # 开始拼接并保存
        print("There are ",len(front_names)," images to be process : please wait...")
        for front_name_,right_name_,left_name_,back_name in zip(front_names,right_names,left_names,back_names):
            front_name = os.path.join(bev_source_img_path,front_name_[:-1])
            right_name = os.path.join(bev_source_img_path,right_name_[:-1])
            left_name  = os.path.join(bev_source_img_path,left_name_[:-1])
            back_name  = os.path.join(bev_source_img_path,back_name[:-1])
            image_save_path = os.path.join(bev_img_path,front_name_[:-1])

            #print(image_save_path)
            global bev_image_count
            if bev_image_count % 100 == 0:
                print(bev_image_count)
            bev_image_count = bev_image_count + 1

            front = cv.imread(front_name)       # 前相机图片
            back = cv.imread(back_name)         # 后相机图片   
            left  = cv.imread(left_name)        # 左相机图片
            right  = cv.imread(right_name)      # 右相机图片

            bevGenerator = BevGenerator(blend=True,balance=True)                # 初始化环视鸟瞰图生成器
            surround = bevGenerator(front,back,left,right,car_img)              # 得到环视鸟瞰图

            cv.imwrite(image_save_path, surround)
            # cv.imshow('surround', surround)
            # cv.waitKey(1)
        print("Remap Completed.")
    else:
        print("Do not need to remap to BEV")

if __name__ == '__main__':
    main()
