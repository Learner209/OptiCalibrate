import pyrealsense2 as rs
import numpy as np
from .base_camera import BaseCamera
import json
from loguru import logger
import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RealsenseConfig:
    serial_number: str
    rgb_resolution: List[int]
    depth_resolution: List[int]
    fps: int
    depth_range: List[float]
    align_frames: bool
    decimate: int
    calibration_file: str

class Realsense(BaseCamera):
    def __init__(self, **kwargs):
       # 将kwargs转换为配置对象
        self.config = RealsenseConfig(**kwargs)
        
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        
        if self.config.serial_number:
            self.rs_config.enable_device(self.config.serial_number)
        
        self.rs_config.enable_stream(rs.stream.color, 
                                   self.config.rgb_resolution[0], 
                                   self.config.rgb_resolution[1], 
                                   rs.format.bgr8, 
                                   self.config.fps)
        
        self.rs_config.enable_stream(rs.stream.depth,
                                   self.config.depth_resolution[0],
                                   self.config.depth_resolution[1],
                                   rs.format.z16,
                                   self.config.fps)

        self.profile = self.pipeline.start(self.rs_config)
        
        # 检查并加载相机内参
        if hasattr(self.config, 'calibration_file') and self.config.calibration_file:
            if os.path.exists(self.config.calibration_file):
                self.load_camera_params(self.config.calibration_file)
            else:
                logger.warning(f"标定文件 {self.config.calibration_file} 未找到")
                self.intrinsics = None
                self.distortion = None
        else:
            logger.warning("未提供标定文件，相机参数可能无法加载")
            self.intrinsics = None
            self.distortion = None

    def load_camera_params(self, calibration_file):
        """
        从标定文件中加载相机参数
        
        参数:
            calibration_file (str): 标定文件路径
        """
        with open(calibration_file, 'r') as f:
            calib_data = json.load(f)
        self.intrinsics = np.array(calib_data['camera_matrix'])
        self.distortion = np.array(calib_data['dist_coeff'])

    def capture_frame(self):
        """
        从相机捕获一帧图像
        
        返回值:
            tuple: 包含 (color_frame, depth_frame) 的元组,两者均为numpy数组
                  如果捕获失败则返回 (None, None)
        """
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None
            return np.asanyarray(color_frame.get_data()), np.asanyarray(depth_frame.get_data())
        except Exception as e:
            logger.error(f"RealSense error: {str(e)}")
            return None, None

    def get_intrinsics(self):
        """
        获取相机内参矩阵
        
        返回值:
            numpy.ndarray: 相机内参矩阵
        """
        return self.intrinsics

    def get_distortion(self):
        """
        获取相机畸变系数
        
        返回值:
            numpy.ndarray: 畸变系数数组
        """
        return self.distortion

    def __del__(self):
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop()