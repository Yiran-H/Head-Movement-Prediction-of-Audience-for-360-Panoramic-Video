# -*- coding: utf-8 -*-

#坐标转换计算工具包
import math
from video import *
import numpy as np


'''
# 统计当前yaw、pitch所对应的视口内的tile_id序列
class Tile_counter:
    def counter(self, yaw, pitch):
        fov_half_length = 100.0 / 360.0 * 2 * math.pi / 2  # FoV视口覆盖范围的一半长度
        reader = Video()
        max_x = reader.max_x
        max_y = reader.max_y
        arr = []
        yaw_radians = yaw / 180.0 * math.pi
        pitch_radians = pitch / 180.0 * math.pi
        for i in range(0, max_y):
            for j in range(0, max_x):
                tile_longitude = ((j + 0.5) / max_x - 0.5) * 360
                tile_latitude = (0.5 - (i + 0.5) / max_y) * 180
                tile_longitude_radians = tile_longitude / 180.0 * math.pi
                tile_latitude_radians = tile_latitude / 180.0 * math.pi
                current_distence = math.acos(math.cos(pitch_radians) * math.cos(tile_latitude_radians) * math.cos(yaw_radians - tile_longitude_radians) + math.sin(pitch_radians) * math.sin(tile_latitude_radians))
                tile_index = i * max_x + j + 1
                if current_distence <= fov_half_length:
                    arr.append(tile_index)
        return arr
'''

# 统计当前yaw、pitch所对应tile_id
class Transform_tile:
    def __init__(self):
        self.video = Video()
        self.configure = Configuration()
        self.video_name = self.video.video_name
        self.predict_time = self.configure.predict_time
        self.max_x = self.video.max_x
        self.max_y = self.video.max_y

        self.width = self.video.width
        self.height = self.video.height

        # 算出单个tile的角度跨度
        self.tile_width = 360 / self.max_x
        self.tile_height = 180 / self.max_y

    #没用了，改进版是tile_counter
    def counter(self, yaw, pitch):
        fov_half_length = 100.0 / 360.0 * 2 * math.pi / 2  # FoV视口覆盖范围的一半长度
        max_x = self.max_x
        max_y = self.max_y
        arr = []
        center_index = []
        yaw_radians = yaw / 180.0 * math.pi
        pitch_radians = pitch / 180.0 * math.pi
        for i in range(0, max_y):
            for j in range(0, max_x):
                tile_longitude = ((j + 0.5) / max_x - 0.5) * 360
                tile_latitude = (0.5 - (i + 0.5) / max_y) * 180
                tile_longitude_radians = tile_longitude / 180.0 * math.pi
                tile_latitude_radians = tile_latitude / 180.0 * math.pi
                current_distence = math.acos(math.cos(pitch_radians) * math.cos(tile_latitude_radians) * math.cos(yaw_radians - tile_longitude_radians) + math.sin(pitch_radians) * math.sin(tile_latitude_radians))
                tile_index = i * max_x + j + 1
                if current_distence <= fov_half_length:
                    arr.append(tile_index)
        center_index.append(self.transform_coordinates(yaw, pitch))
        result = list(set(arr).union(set(center_index)))
        return result


    def tile_counter(self, yaw, pitch):
        fov_half_length = 100.0 / 360.0 * 2 * math.pi / 2  # FoV视口覆盖范围的一半长度
        max_x = self.max_x
        max_y = self.max_y
        yaw_radians = yaw / 180.0 * math.pi
        pitch_radians = pitch / 180.0 * math.pi
        edge_index = []
        center_index = []
        for i in range(max_y):
            for j in range(max_x):
                for k in range(4):  # 每个tile取4个顶点判断
                    tile_longitude = None
                    tile_latitude = None
                    if k == 0:
                        tile_longitude = (j / max_x - 0.5) * 360
                        tile_latitude = (0.5 - i / max_y) * 180
                    elif k == 1:
                        tile_longitude = ((j + 1) / max_x - 0.5) * 360
                        tile_latitude = (0.5 - i / max_y) * 180
                    elif k == 2:
                        tile_longitude = (j / max_x - 0.5) * 360
                        tile_latitude = (0.5 - (i + 1) / max_y) * 180
                    elif k == 3:
                        tile_longitude = ((j + 1) / max_x - 0.5) * 360
                        tile_latitude = (0.5 - (i + 1) / max_y) * 180
                    tile_longitude_radians = tile_longitude / 180.0 * math.pi
                    tile_latitude_radians = tile_latitude / 180.0 * math.pi
                    current_distence = math.acos(math.cos(pitch_radians) * math.cos(tile_latitude_radians) * math.cos(yaw_radians - tile_longitude_radians) + math.sin(pitch_radians) * math.sin(tile_latitude_radians))  # 当前tile的某个点到注视点的距离
                    tile_index = i * max_x + j + 1
                    if current_distence <= fov_half_length:
                        edge_index.append(tile_index)
        center_index.append(self.transform_coordinates(yaw, pitch))
        result = list(set(edge_index).union(set(center_index)))
        return result


    # 坐标转换,yaw、pitch转换为tile映射
    def transform_coordinates(self, yaw, pitch):
        #print('in', yaw, pitch)
        x, y = self.to_coordinates(yaw, pitch)
        tile_no = int(self.to_tile(x, y))
        return tile_no

    # yaw、pitch转标准矩形坐标（360*180的角度）
    def to_coordinates(self, in_yaw, in_pitch):
        out_x = in_yaw + 180
        out_y = 90 - in_pitch
        return out_x, out_y

    def to_pixel_coordinate(self, in_yaw, in_pitch):
        x = int(((in_yaw + 180) / 360) * self.width)
        y = int(((90 - in_pitch) / 180) * self.height)
        return x, y

    # 标准矩形坐标（360*180的角度）转tile映射
    def to_tile(self, in_x, in_y):
        i = None
        j = None
        # 这里有问题，回头仔细想下tile映射
        if 0 <= in_x < 360 and 0 <= in_y < 180:
            i = in_y // self.tile_height
            j = in_x // self.tile_width + 1
        elif in_x == 360 and 0 <= in_y < 180:
            i = in_y // self.tile_height
            j = in_x // self.tile_width
        elif 0 <= in_x < 360 and in_y == 180:
            i = in_y // self.tile_height - 1
            j = in_x // self.tile_width + 1
        elif in_x == 360 and in_y == 180:
            i = in_y // self.tile_height - 1
            j = in_x // self.tile_width
        else:
            print("ERROR!!原因：坐标转换出错！！")
            os._exit(0)
        tile_no = i * self.max_x + j
        #print(i, j, tile_no)
        return tile_no

    # 根据帧号计算tile中心坐标
    def cal_tile_center(self, tile_no):
        #print(tile_no)
        index = tile_no - 1
        x = index % self.max_x
        y = index // self.max_x
        center_yaw = (x * self.tile_width + self.tile_width / 2) - 180
        center_pitch = 90 - (y * self.tile_height + self.tile_height / 2)
        return center_yaw, center_pitch

    def cal_tile_index(self, tile_no):
        #print(tile_no)
        index = tile_no - 1
        j = index % self.max_x
        i = index // self.max_x
        return i, j

    def corrected_coordinates(self, coordinates):
        frame = coordinates[0]
        yaw = coordinates[1]
        pitch = coordinates[2]
        tile_no = self.transform_coordinates(yaw, pitch)
        center_yaw, center_pitch = self.cal_tile_center(tile_no)
        result = [frame, center_yaw, center_pitch]
        return result

    #角速度笔记：计算三角函数要转弧度制
    # 计算大圆距离
    def cal_GreatCircle_distance(self, euler_from, euler_to):
        pitch_from = np.radians(euler_from[1])  # B1
        pitch_to = math.radians(euler_to[1])  # B2
        delta_yaw = math.radians(euler_to[0] - euler_from[0])  # B2 - B1
        cos = math.sin(pitch_from) * math.sin(pitch_to) + math.cos(pitch_from) * math.cos(pitch_to) * math.cos(delta_yaw)
        distance = abs(math.degrees(math.acos(cos)))
        return distance

    # 计算欧拉角距离
    def cal_euler_distance(self, euler_from, euler_to):
        coordinate_from = self.euler_to_xyz(euler_from)
        coordinate_to = self.euler_to_xyz(euler_to)
        dot = coordinate_from[0] * coordinate_to[0] + coordinate_from[1] * coordinate_to[1] + coordinate_from[2] * coordinate_to[2]

        length_from = math.sqrt(pow(coordinate_from[0], 2) + pow(coordinate_from[1], 2) + pow(coordinate_from[2], 2))
        length_to = math.sqrt(pow(coordinate_to[0], 2) + pow(coordinate_to[1], 2) + pow(coordinate_to[2], 2))
        cos = dot / (length_from * length_to)
        # 检测浮点数错误
        if cos > 1:
            print('浮点数警告！', cos, '坐标为', euler_from, euler_to)
            cos = 1
        elif cos < -1:
            print('浮点数警告！', cos, '坐标为', euler_from, euler_to)
            cos = -1
        included_angle = abs(math.degrees(math.acos(cos)))
        return included_angle

    def euler_to_xyz(self, euler):
        cal_yaw = math.radians(-euler[0])
        cal_pitch = math.radians(euler[1])
        x = math.cos(cal_yaw) * math.cos(cal_pitch)
        y = math.sin(cal_yaw) * math.cos(cal_pitch)
        z = math.sin(cal_pitch)
        xyz = (x, y, z)
        return xyz



'''
class Tile_counter:
    def counter(self, yaw, pitch):
        fov_half_length = 100.0 / 360.0 * 2 * math.pi / 2  # FoV视口覆盖范围的一半长度
        reader = Video()
        max_x = reader.max_x
        max_y = reader.max_y
        arr = array.array('i')
        yaw_radians = yaw / 180.0 * math.pi
        pitch_radians = pitch / 180.0 * math.pi
        for i in range(0, max_y):
            for j in range(0, max_x):
                tile_longitude = ((j + 0.5) / 16 - 0.5) * 360
                tile_latitude = (0.5 - (i + 0.5) / 8) * 180
                tile_longitude_radians = tile_longitude / 180.0 * math.pi
                tile_latitude_radians = tile_latitude / 180.0 * math.pi
                current_distence = math.acos(math.cos(pitch_radians) * math.cos(tile_latitude_radians) * math.cos(yaw_radians - tile_longitude_radians) + math.sin(pitch_radians) * math.sin(tile_latitude_radians))
                tile_index = i * 16 + j + 1
                if current_distence <= fov_half_length:
                    arr.append(tile_index)
        return arr.tolist()
'''


if __name__ == '__main__':
    a = Transform_tile().cal_euler_distance([-10, 0], [45, 45])
    b = Transform_tile().cal_GreatCircle_distance([-10, 0], [45, 45])
    print(a, b)
    #b = Transform_tile().tile_counter(2.0346635560358095, 5.9907504035225978)
    #print(b)




