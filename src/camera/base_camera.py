from abc import ABC, abstractmethod
from omegaconf import DictConfig
import numpy as np

class BaseCamera(ABC):

    def __init__(self, config: DictConfig):
       self.config = config 

    @abstractmethod
    def capture_frame(self):
        """
        从相机捕获一帧图像

        返回值:
            tuple: 包含 (color_frame, depth_frame) 的元组,两者均为numpy数组
                  如果捕获失败则返回 (None, None)
        """
        pass

    @abstractmethod
    def get_intrinsics(self):
        """
        获取相机内参矩阵

        返回值:
            numpy.ndarray: 相机内参矩阵
        """
        pass

    @abstractmethod
    def get_distortion(self):
        """
        获取相机畸变系数

        返回值:
            numpy.ndarray: 畸变系数数组
        """
        pass

    @abstractmethod
    def load_camera_params(self, calibration_file: str):
        """
        从标定文件中加载相机参数

        参数:
            calibration_file (str): 标定文件路径
        """
        pass