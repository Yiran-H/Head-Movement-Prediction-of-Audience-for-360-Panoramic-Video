from database_player import HMDplayer
import matplotlib.pyplot as plt  # plt 用于显示图片
from collections import deque
from pylab import mpl
from predictpolicy import *
import pandas as pd
import saliency_policy
import video
from transformtile import Transform_tile
import csv
import os
# 设置字体（不然显示不出中文）
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# 输入用户名即可
no_user = 1

# 路径
file_path = video.Video().test_set_path
video_name = video.Video().video_name
file_name = 'user_' + "%03d" % no_user + '_' + video_name + '.csv'
file_in = os.path.join(file_path, file_name)

class Init():
    def __init__(self):
        self.transform_tile = Transform_tile()
        self.frame_no = deque()
        self.yaw = deque()
        self.pitch = deque()
        self.execute()

    def execute(self):
        df = pd.read_csv(file_in, usecols=[0, 1, 2])
        data = np.asarray(df)
        for i in range(len(data)):
            #coordinates = self.transform_tile.corrected_coordinates([i + 1, data[i][1], data[i][2]])# tile为中心
            coordinates = [i + 1, data[i][1], data[i][2]]  # 坐标为中心
            frame_no = coordinates[0]
            # print("当前帧为", frame)
            yaw = coordinates[1]
            pitch = coordinates[2]
            self.frame_no.append(frame_no)
            self.yaw.append(yaw)
            self.pitch.append(pitch)


class Policy:
    def __init__(self, predict_policy):
        self.policy = predict_policy
        self.name = self.policy().name
        print(self.name)
        self.yaw = deque()
        self.pitch = deque()
        self.tile_accuracy = None
        self.execute()


    def execute(self):
        play = HMDplayer(file_in, self.policy)
        predict = play.pre_coordinates
        for elem in predict:
            yaw = [elem[0], elem[1]]
            pitch = [elem[0], elem[2]]
            self.yaw.append(yaw)
            self.pitch.append(pitch)
        self.tile_accuracy = play.Mean_accuracy



class Saliency_Policy():
    def __init__(self):
        self.name = 'Saliency_Policy'
        self.yaw = deque()
        self.pitch = deque()
        self.tile_accuracy = None
        self.execute()

    def execute(self):
        play = saliency_policy.Player(file_in)
        predict = play.pre_coordinates
        for elem in predict:
            yaw = [elem[0], elem[1]]
            pitch = [elem[0], elem[2]]
            self.yaw.append(yaw)
            self.pitch.append(pitch)
        self.tile_accuracy = play.Mean_accuracy


'''
    def write(self):
        print("正在写入...")
        csv_name = "out_" + str(Configuration().predict_time) + ".csv"
        out = open(csv_name, 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(['', '精确度', 'tile精确度'])
        csv_write.writerow([self.policy().name, self.accuracy, self.tile_accuracy])
        out.close()
        print("写入成功！")
'''

class Analyst:
    def __init__(self):
        self.label = []
        self.real_yaw = deque()
        self.real_pitch = deque()
        self.predict_yaw = deque()
        self.predict_pitch = deque()
        self.accuracy = []
        self.colors = ['red', 'orange', 'green', 'purple', 'yellow']  # 设置颜色
        self.run()
        self.map()

    def run(self):
        # 再设置策略
        #self.result.append(Saliency_Policy().tile_accuracy)
        init = Init()
        self.frame_no = init.frame_no
        self.real_yaw = init.yaw
        self.real_pitch = init.pitch
        # 策略列表
        policy = [Policy(LSRpolicy), Policy(LRpolicy), Policy(TLPpolicy)]
        for elem in policy:
            self.predict_yaw.append(elem.yaw)
            self.predict_pitch.append(elem.pitch)
            self.accuracy.append(elem.tile_accuracy)
            self.label.append(elem.name)
        for i in range(len(policy)):
            print(self.label[i], self.accuracy[i])



    def map(self):
        # 一些初始化
        plt.figure(1)
        yaw = plt.subplot(2, 1, 1)
        pitch = plt.subplot(2, 1, 2)


        plt.sca(yaw)  # 选中yaw图
        plt.title("yaw的运动轨迹")
        plt.xlim(0, Video().video_frames)
        plt.ylim(-180, 180)
        real_frame = self.frame_no
        real_yaw = self.real_yaw
        plt.scatter(real_frame, real_yaw, s=1, color='black', label='real')
        plt.xlabel("帧")
        plt.ylabel("角度")

        # plt.xlim(0.8, 1) # X轴上下界必须再商量
        #plt.ylim(0, 1)
        for i in range(len(self.predict_yaw)):
            frame = np.array(self.predict_yaw[i])[:, 0]
            data = np.array(self.predict_yaw[i])[:, 1]
            plt.scatter(frame, data, s=1, color=self.colors[i], label=self.label[i])
        plt.legend(loc="best")
        #plt.grid(linestyle='--')  # 设置网格

        # 选中pitch图
        plt.sca(pitch)
        plt.title("pitch的运动轨迹")
        plt.xlim(0, Video().video_frames)
        plt.ylim(-90, 90)
        real_frame = self.frame_no
        real_pitch = self.real_pitch
        plt.scatter(real_frame, real_pitch, s=1, color='black', label='real')
        plt.xlabel("帧")
        plt.ylabel("角度")

        for i in range(len(self.predict_pitch)):
            frame = np.array(self.predict_pitch[i])[:, 0]
            data = np.array(self.predict_pitch[i])[:, 1]
            plt.scatter(frame, data, s=1, color=self.colors[i], label=self.label[i])
        plt.legend(loc="best")
        plt.show()


if __name__ == '__main__':
    Analyst()



