import cv2
import numpy as np
from .base_marker import BaseMarker
from typing import Optional, Tuple, Any, Dict
from dataclasses import dataclass
from loguru import logger

@dataclass
class ArucoConfig:
    dictionary: str
    marker_size: float
    parameters: Dict[str, float]

class Aruco(BaseMarker):
    def __init__(self, **kwargs):
        # 创建配置对象并传递给父类
        config = ArucoConfig(**kwargs)
        super().__init__(config)
        
        try:
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
                else:
                    logger.warning(f"未知的参数: {param_name}")
            
            self.marker_size = self.config.marker_size
            self.rep_error = None
            
            logger.info("ArUco标记检测器初始化完成")
            
        except Exception as e:
            logger.error(f"ArUco初始化失败: {str(e)}")
            raise

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
        try:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                image, self.aruco_dict, parameters=self.parameters)
            return corners, ids, rejected
        except Exception as e:
            logger.error(f"ArUco检测失败: {str(e)}")
            return None, None, None

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
        try:
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
            
        except Exception as e:
            logger.error(f"ArUco位姿估计失败: {str(e)}")
            return None, None, None

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
        try:
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
            
        except Exception as e:
            logger.error(f"ArUco结果绘制失败: {str(e)}")
            return image

    def generate_marker(self, marker_id: int, size: int = 200) -> Optional[np.ndarray]:
        """
        生成ArUco标记图像
        Args:
            marker_id: 标记ID
            size: 图像大小
        Returns:
            生成的标记图像
        """
        try:
            marker_image = np.zeros((size, size), dtype=np.uint8)
            cv2.aruco.drawMarker(self.aruco_dict, marker_id, size, marker_image, 1)
            return marker_image
        except Exception as e:
            logger.error(f"ArUco标记生成失败: {str(e)}")
            return None