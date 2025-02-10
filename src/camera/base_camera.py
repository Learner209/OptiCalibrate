from abc import ABC, abstractmethod
from omegaconf import DictConfig
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


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

    # @abstractmethod
    # def capture_point_cloud(self):
    #     """
    #     捕获点云数据
    #
    #     Returns:
    #         Tuple[np.ndarray, np.ndarray]: (points, color_image)
    #             points: Nx3 点云数据
    #             color_image: 对应的彩色图像
    #     """
    #     pass


@dataclass
class ValidationResult:
    """验证结果数据类"""

    success: bool
    error_stats: Optional[Dict] = None
    points_base: Optional[np.ndarray] = None
    color_image: Optional[np.ndarray] = None  # 添加颜色图像字段


@dataclass
class ValidatorConfig:
    """验证器配置数据类"""

    # 验证参数
    min_points: int = 1000
    max_distance: float = 1.5  # 降低最大距离阈值
    max_rmse: float = 0.02  # 增加RMSE容差

    # 点云处理参数
    enable_cuda: bool = False
    voxel_size: float = 0.005
    remove_outliers: bool = True
    outlier_std: float = 3.0  # 增加离群点阈值
    outlier_nb_neighbors: int = 30  # 减少邻居点数量
