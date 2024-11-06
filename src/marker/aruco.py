import cv2
import numpy as np
from .base_marker import BaseMarker
from omegaconf import DictConfig
from typing import Optional, Tuple, Any, Dict
from dataclasses import dataclass

@dataclass
class ArucoConfig:
    dictionary: str
    marker_size: float
    parameters: Dict[str, float]

class Aruco(BaseMarker):
    def __init__(self, **kwargs):
        # 将kwargs转换为配置对象
        self.config = ArucoConfig(**kwargs)
        
        # 获取预定义的字典
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, self.config.dictionary)
        )
        # 创建检测器参数
        self.parameters = cv2.aruco.DetectorParameters()
        
        # 设置检测器参数
        for param_name, param_value in self.config.parameters.items():
            if hasattr(self.parameters, param_name):
                setattr(self.parameters, param_name, param_value)
        
        self.marker_size = self.config.marker_size
        self.rep_error = None

    def detect(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Any]:
        """
        检测图像中的ArUco标记
        Args:
            image: 输入图像
        Returns:
            corners: 角点坐标
            ids: 标记ID
            rejected: 被拒绝的候选区域
        """
        corners, ids, rejected = cv2.aruco.detectMarkers(
            image, self.aruco_dict, parameters=self.parameters)
        return corners, ids, rejected

    def estimate_pose(self, corners: np.ndarray, ids: np.ndarray,
                 camera_matrix: np.ndarray, 
                 dist_coeffs: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        估计标记的位姿
        """
        if ids is None or len(ids) == 0:
            return None, None, None
            
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, camera_matrix, dist_coeffs)
        
        if rvecs is not None:
            # 计算所有标记的平均重投影误差
            self.rep_error = np.mean([
                np.sqrt(np.mean((corners[i].reshape(-1, 2) - 
                    cv2.projectPoints(self.marker_size * np.array([[-0.5, 0.5, 0], 
                    [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]], dtype=np.float32),
                    rvecs[i], tvecs[i], camera_matrix, dist_coeffs)[0].reshape(-1, 2)) ** 2))
                for i in range(len(ids))
            ])
            return rvecs, tvecs, self.rep_error
        
        return None, None, None 

    def draw_results(self, image: np.ndarray, corners: np.ndarray, ids: np.ndarray,
                rvecs: Optional[np.ndarray] = None, tvecs: Optional[np.ndarray] = None,
                camera_matrix: Optional[np.ndarray] = None, 
                dist_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        绘制检测和位姿估计结果
        """
        if ids is None or len(ids) == 0:
            return image
        
        img = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
        if rvecs is not None and camera_matrix is not None:
            for rvec, tvec in zip(rvecs, tvecs):
                img = cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, 
                                    rvec, tvec, self.marker_size/2)
                
            if self.rep_error is not None:
                # 根据误差大小选择颜色
                if self.rep_error < 1:
                    color = (0, 255, 0)  # 绿色：优秀
                elif self.rep_error < 2:
                    color = (0, 255, 255)  # 黄色：良好
                elif self.rep_error < 3:
                    color = (0, 165, 255)  # 橙色：可接受
                else:
                    color = (0, 0, 255)  # 红色：需改进
                    
                cv2.putText(img, f"RMSE: {self.rep_error:.4f}px", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img

    def generate_marker(self, marker_id: int, size: int = 200) -> np.ndarray:
        """生成ArUco标记图像"""
        marker_image = np.zeros((size, size), dtype=np.uint8)
        cv2.aruco.drawMarker(self.aruco_dict, marker_id, size, marker_image, 1)
        return marker_image