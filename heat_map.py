# 生成热力图
import matplotlib.pyplot as plt  # plt 用于显示图片
from scipy.spatial import distance
from colour import Color
from transformtile import Transform_tile
from collections import deque  # 双端队列
from pyheatmap.heatmap import HeatMap
from pylab import mpl
from video import *
import numpy as np
import pandas as pd
import imutils
import math
import cv2
import glob
import os
import heatmap
# 设置字体（不然显示不出中文）
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 缓冲区路径
cache_path = "heatmap"

class Heatmap_maker:
    def __init__(self):
        # 视频信息初始化
        self.video = Video()  # 视频属性读取（见video.py）
        # self.configure = Configuration()
        self.tf = Transform_tile()  # 坐标换算工具包（见transform_tile.py）
        self.video_name = self.video.video_name  # 视频名称
        self.dataset_path = self.video.dataset_path  # 数据集路径
        self.width = self.video.width  # 视频宽
        self.height = self.video.height  # 视频高
        self.radius = int(math.sqrt((self.width * self.height) * 0.001 / math.pi))  # 热力图的点半径
        print('半径是', self.radius)
        self.frames = self.video.video_frames  # 总帧数
        self.fps = self.video.video_fps  # 帧率
        self.counter = self.user_counter()  # 下文定义的方法，统计用户数
        self.size = (self.width, self.height)  # 热力图宽高比

        self.saliency_data = deque()  
        # 存放用户位置，二维，每行对应一个用户，每列对应同一帧的数据。但是每一个元素都是一个像素坐标（x, y)

        #热力图相关

        self.cache_path = os.path.join('heatmap', self.video.video_name)  # 热力图视频生成位置
        self.frame_path = os.path.join(self.cache_path, "frame")  # 热力图帧生成位置

        self.dataset_reader()  # 开始读数据（下文定义的方法）
        self.videoMaker()  # 视频制作（下文定义的方法）


    # 读取数据
    def dataset_reader(self):
        files = os.listdir(self.dataset_path)  # 遍历数据集文件夹里的所有文件名
        # print(files)
        for file in files:  # 对每个文件名进行循环
            print('正在读取：', file, '...')
            csv = os.path.join(self.dataset_path, file)  # 读取当前文件的文件名
            self.saliency_data.append(self.user_player(csv))  # 把这个用户的数据入队（user_player()用来读取一个用户数据，下文有定义）
        print('读取完毕！')
    # 播放单个用户数据
    def user_player(self, path):
        df = pd.read_csv(path, usecols=[0, 1, 2])  # 读取单个用户数据文件
        raw_data = np.array(df)  # 存放初始数据
        cal_data = deque(maxlen=len(raw_data))  # 存放转换坐标后的数据
        for i in range(len(raw_data)):  # 对所有的坐标进行像素坐标转换（yaw、pitch）转（x，y）
            x, y = self.tf.to_pixel_coordinate(raw_data[i][1], raw_data[i][2])  # 转换像素坐标
            cal_data.append([x, y])  # 数据入队
        return cal_data  # 返回给上层

    def user_counter(self):  # 统计用户总人数，就是遍历下文件夹里多少文件
        path_name = os.path.join(self.dataset_path, '*.csv')
        path_file_number = glob.glob(path_name)  # 获取当前文件夹下个数
        return len(path_file_number)

    # 路径初始化，如果路径不存在，新建文件夹路径
    def init_map_path(self):
        print("初始化缓冲区...")
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)
        print("初始化路径...")
        if not os.path.exists(self.frame_path):
            os.mkdir(self.frame_path)
        print("初始化成功！")

    # 制作热力图帧
    def frame_maker(self):
        for i in range(0, self.frames):  # 从第一帧开始，i指的是帧的数组下标，从0开始
            print("正在生成第", i + 1, "张图......")
            frame = []  # 当前帧的数据初始化
            for j in range(len(self.saliency_data)):  # 因为同一列对应同一帧的数据，这里所以要找不同行但是同列的数据
                frame.append(self.saliency_data[j][i])
            #print(frame)
            # 这下面就是套路，详情看pyheatmap，搜github就有
            hm = HeatMap(frame, width=self.width, height=self.height)  # 把这一帧所有用户的数据（二维列表），width和heat对应这张图片的尺寸，要严格和视频帧尺寸相同！
            image = str(i + 1) + ".png"  # 热力图图片序号（对应每帧）
            image_dir = os.path.join(self.frame_path, image)  # 图片路径
            hm.heatmap(save_as=image_dir, r=self.radius)  # 保存
            print("图片生成成功！")

    # 视频开始制作
    def videoMaker(self):  # path:热力图路径
        self.init_map_path()  # 初始化文件路径
        self.frame_maker()  # 先生成热力图帧，帧生成好了再进行视频渲染
        print("初始化视频生成器路径...")
        if not os.path.exists(self.frame_path):  # 这里检查热力图是否存在，不存在直接退出
            print("初始化失败！没有图片路径！")
            os._exit(1)

        print("初始化视频生成器成功！")
        video_path = os.path.join(self.cache_path, self.video.video_name + '.avi')  # 输出视频视频路径
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, self.size)  # 定义视频输出格式、名称
        print("视频生成中...")
        # 视频初始化
        video_in = self.video.video_path  # 原始文件名.MP4
        camera = cv2.VideoCapture(video_in)  # 用opencv读取原始视频

        # 开始播放每一帧，并写入
        for i in range(0, self.frames):  # 修改： 基于总帧数的生成方式
            success, frame = camera.read()  # 读取原始视频帧，success判断是否读取成功，frame指的是读取的原始视频帧
            print('现在是第', i + 1, '帧')
            if not success:  # 如果读取帧失败，结束循环
                break
            heat_map_name = str(i + 1) + ".png"  # 热力图名称（图片名从1开始，但是i从0开始，这里对应好）
            heat_map_dir = os.path.join(self.frame_path, heat_map_name)  # 热力图路径
            heat_map = cv2.imread(heat_map_dir, 1)  # 读取此热力图
            alpha = 0.5  # 设置覆盖图片的透明度（两张图片重叠，这里透明图各一半）
            overlay = frame.copy()  # 复制帧，这里要不要貌似都无所谓，之前复制别人博主代码的，人家这么写的，无所谓
            #cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), -1)  # 设置蓝色为热度图基本色，这里不要
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # 将背景热度图覆盖到原图
            cv2.addWeighted(heat_map, alpha, frame, 1 - alpha, 0, frame)  # 将热度图覆盖到原图

            if cv2.waitKey(100) & 0xff == ord('q'):  # 这里是opencv播放视频时的一些固有的代码，别问，复制就好了，大概意思是按q键提前结束播放
                break
            video.write(frame)  # 写入帧
            frame = imutils.resize(frame, width=1000)  # 这里是调整你播放窗口尺寸的，因为全景视频原尺寸播放，整个屏幕装不下，所以调整播放窗口大小用的
            cv2.imshow('frame', frame)  # 播放窗口显示出来
        video.release()  # 结束，释放视频
        cv2.destroyAllWindows()  # 释放窗口
        print("视频生成成功！")




'''

class Heatmap_maker:
    def __init__(self):
        self.video = Video()
        print("初始化缓冲区...")
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        self.cache = cache_path
        print("缓冲区初始化成功！")
        self.size = (640, 480)

    def iniCache(self):
        print("初始化路径...")
        dir_path = os.path.join(self.cache, "Thermal_map")
        file = os.path.join(dir_path, self.video.video_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if not os.path.exists(file):
            os.mkdir(file)
        print("初始化成功！")
        dr = Probability_matrix()
        numpy = dr.analyst()
        for i in range(0, self.video.video_frames):
            print("正在生成第", i + 1, "张图......")
            temp = numpy[i]
            frame = np.reshape(temp, (self.video.max_y, self.video.max_x))
            plt.imshow(frame, interpolation='nearest', cmap=plt.cm.hot, vmin=0, vmax=1, extent=(0, self.video.max_x, 0, self.video.max_y))  # cmap=plt.cm.gray绘制黑白图像
            # plt.colorbar()
            image = str(i + 1) + ".png"
            image_dir = os.path.join(file, image)
            plt.savefig(image_dir)
            print("图片生成成功！")
            plt.close()
        return file
    #  视频生成器
    def videoMaker(self, path):  # path:热力图路径
        print("初始化视频生成器路径...")
        dir_path = os.path.join(self.cache, 'video')
        file_path = os.path.join(dir_path, self.video.video_name)
        if not os.path.exists(path):
            print("初始化失败！没有图片路径！")
            os._exit(1)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        print("初始化视频生成器成功！")
        video_path = os.path.join(file_path, self.video.video_name + '.avi')
        video = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*'DIVX'), self.video.video_fps, self.size)
        print("视频生成中...")
        for i in range(0, self.video.video_frames):  # 修改： 基于总帧数的生成方式
            img_name = str(i + 1) + ".png"
            img_dir = os.path.join(path, img_name)
            img = cv.imread(img_dir, 1)
            video.write(img)
        video.release()
        cv.destroyAllWindows()
        print("视频生成成功！")
        return video_path
    # 播放器初始化
    def iniPlayer(self):
        print("初始化播放器...")
        path = self.iniCache()
        video_path = self.videoMaker(path)
        print("播放器初始化成功！")
        return video_path
    # 播放功能
    def play(self, path):
        cap = cv.VideoCapture(path)
        dt = int(1 / self.video.video_fps * 1000)
        n = 1
        print("开始播放，按下q提前结束播放...")
        while cap.isOpened():
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("视频播放完毕！")
                break
            cv.imshow('video', frame)
            if cv.waitKey(dt) == ord('q'):
                break
            n += 1
        cap.release()
        cv.destroyAllWindows()

    def autoplay(self):
        video = self.iniPlayer()
        self.play(video)
'''
if __name__ == '__main__':
    Heatmap_maker()

'''
class Dataset_reader:
    def __init__(self):
        print('初始化用户数据...')
        self.video = Video()
        self.config = Configuration()
        self.transform_tile = Transform_tile()
        frames = self.video.video_frames
        self.saliency_set = deque(maxlen=frames)
        current_frame_history = deque()
        for i in range(frames):
            elem = [i + 1, current_frame_history]
            self.saliency_set.append(elem)
        # print(self.saliency_set)
        path = self.video.dataset_path
        files = os.listdir(path)
        for file in files:
            # print('正在读取：', file)
            csv_path = os.path.join(path, file)
            file = pd.read_csv(csv_path, usecols=[0, 1, 2])
            data = np.array(pd.DataFrame(file))
            for i in range(len(data)):
                # 原始yaw、pitch
                # coordinate = (data[i][1], data[i][2])
                # 像素坐标
                x, y = self.transform_tile.to_pixel_coordinate(data[i][1], data[i][2])
                coordinate = (x, y)

                temp = deque(self.saliency_set[i][1])
                temp.append(coordinate)
                self.saliency_set[i][1] = temp
        # for i in range(len(self.saliency_set)):
        #    print(len(self.saliency_set[i][1]))

    def play(self):
        video_in = self.video.video_path
        camera = cv2.VideoCapture(video_in)
        size = (int(self.video.width), int(self.video.height))
        print(size)
        frame_no = 0
        success = True
        while success:
            success, frame = camera.read()
            saliency_map = self.saliency_set[frame_no][1]
            frame_no += 1

            heatmap = cv2.imread(self.density_heatmap(size, saliency_map, radias=100))
            overlay = frame.copy()
            alpha = 0.5  # 设置覆盖图片的透明度
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), -1)  # 设置蓝色为热度图基本色
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # 将背景热度图覆盖到原图
            cv2.addWeighted(heatmap, alpha, frame, 1 - alpha, 0, frame)  # 将热度图覆盖到原图
            # 限制窗口大小
            frame = imutils.resize(frame, width=500)
            print('show')

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break


    def density_heatmap(self, size, box_centers, radias=100):

        density_range = 100
        gradient = np.linspace(0, 1, density_range)
        img_width = size[0]
        img_height = size[1]
        density_map = np.zeros((img_height, img_width))
        color_map = np.empty([img_height, img_width, 3], dtype=int)
        # get gradient color using rainbow
        cmap = plt.get_cmap("rainbow")  # 使用matplotlib获取颜色梯度
        blue = Color("blue")  # 使用Color来生成颜色梯度
        hex_colors = list(blue.range_to(Color("red"), density_range))
        rgb_colors = [[rgb * 255 for rgb in color.rgb] for color in hex_colors][::-1]
        for i in range(img_height):
            for j in range(img_width):
                for box in box_centers:
                    #print(box)
                    dist = distance.euclidean(box, (j, i))
                    if dist <= radias * 0.25:
                        density_map[i][j] += 10
                    elif dist <= radias:
                        density_map[i][j] += (radias - dist) / (radias * 0.75) * 10
                ratio = min(density_range - 1, int(density_map[i][j]))
                for k in range(3):
                    # color_map[i][j][k] = int(cmap(gradient[ratio])[:3][k]*255)
                    color_map[i][j][k] = rgb_colors[ratio][k]
        return color_map
'''



    #a.autoplay()

# 手动播放
    #path = a.iniCache()
    #a.videoMaker("E:\My_projects\VRplayer\cache\Thermal_map\Roller_coaster_360")
    #a.play("E:\\My_projects/VRplayer/cache/video/Roller_coaster_360/Roller_coaster_360.avi")



