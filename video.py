# -*- coding: utf-8 -*-
import configparser
import cv2
import os
#ini文件路径
file_in = "dash360.ini"

class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance


class Video(Singleton):
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(file_in)
        # ini文件里的common属性
        self.total_layers = config.getint("common", "total_layers")
        self.duration = config.getfloat("common", "duration")
        self.buffer_size = config.getint("common", "buffer_size")
        self.max_x = config.getint("common", "max_x")
        self.max_y = config.getint("common", "max_y")
        self.total_tile_num = config.getint("common", "total_tile_num")
        self.segment_num = config.getint("common", "segment_num")
        self.schedule_policy = config.get("common", "schedule_policy")
        self.server_ip = config.get("common", "server_ip")
        # ini文件里的path属性
        self.dataset_path = config.get("path", "dataset_path")
        self.mpd_base_path = config.get("path", "mpd_base_path")
        self.download_save_path = config.get("path", "download_save_path")
        self.log_download_path = config.get("path", "log_download_path")
        self.log_play_path = config.get("path", "log_play_path")
        # ini文件里的video属性
        video_no = config.getint("video", "video_no")
        self.video_name = 'video_' + "%03d" % video_no
        self.video_path = os.path.join('videos', self.video_name + '_360.mp4')
        self.dataset_path = os.path.join('dataset', 'frame', self.video_name)
        self.train_set_path = os.path.join('train_set', 'frame', self.video_name)
        self.test_set_path = os.path.join('test_set', 'frame', self.video_name)
        # 读取视频信息
        camera = cv2.VideoCapture(self.video_path)
        self.video_fps = camera.get(cv2.CAP_PROP_FPS)
        self.video_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.video_frames / self.video_fps
        self.width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ini文件里的play属性
        self.play_update = config.getint("play", "play_update")
        self.play_pause = config.getint("play", "play_pause")

class Configuration(Singleton):
    def __init__(self):
        config = configparser.ConfigParser()
        config.read("config.ini")
        self.training_time = config.getfloat("prediction", "training_time")
        self.predict_time = config.getfloat("prediction", "predict_time")
        self.error_tolerance = config.getfloat("prediction", "error_tolerance")


if __name__ == '__main__':
    a = Video()
    b = Video()
    print(a.video_name)
    print(a is b)
    print(id(a))
    print(id(b))
