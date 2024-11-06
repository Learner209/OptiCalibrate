from abc import ABC, abstractmethod
import numpy as np
from omegaconf import DictConfig
from typing import Optional, Tuple, Any

class BaseMarker(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Any]:
        """
        检测图像中的标记
        Args:
            image: 输入图像
        Returns:
            corners: 角点坐标
            ids: 标记ID
            rejected: 被拒绝的候选区域
        """
        pass

    @abstractmethod
    def estimate_pose(self, corners: np.ndarray, ids: np.ndarray,
                     camera_matrix: np.ndarray, 
                     dist_coeffs: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        估计标记的位姿
        Args:
            corners: 角点坐标
            ids: 标记ID
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
        Returns:
            rvecs: 旋转向量
            tvecs: 平移向量
            rep_error: 重投影误差
        """
        pass

    @abstractmethod
    def draw_results(self, image: np.ndarray, corners: np.ndarray, ids: np.ndarray,
                    rvecs: Optional[np.ndarray] = None, tvecs: Optional[np.ndarray] = None,
                    camera_matrix: Optional[np.ndarray] = None, 
                    dist_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        绘制检测和位姿估计结果
        Args:
            image: 输入图像
            corners: 角点坐标
            ids: 标记ID
            rvecs: 旋转向量
            tvecs: 平移向量
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
        Returns:
            绘制结果图像
        """
        pass