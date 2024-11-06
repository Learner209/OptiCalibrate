import cv2
import numpy as np
from .base_marker import BaseMarker
from omegaconf import DictConfig
from typing import Optional, Tuple, Any
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ChArUcoConfig:
    dictionary: str
    board_size: List[int]
    square_size: float
    marker_size: float
    parameters: Dict[str, float]

class ChArUco(BaseMarker):
    def __init__(self, **kwargs):
        # 将kwargs转换为配置对象
        self.config = ChArUcoConfig(**kwargs)
        
        # 获取预定义的字典
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, self.config.dictionary)
        )
        # 创建ChArUco板
        self.board = cv2.aruco.CharucoBoard(
            tuple(self.config.board_size),  # (squares_x, squares_y)
            self.config.square_size,        # 棋盘格方块大小
            self.config.marker_size,        # ArUco标记大小
            self.aruco_dict
        )
        # 创建检测器参数
        self.parameters = cv2.aruco.DetectorParameters()
        
        # 设置检测器参数
        for param_name, param_value in self.config.parameters.items():
            if hasattr(self.parameters, param_name):
                setattr(self.parameters, param_name, param_value)
        
        self.rep_error = None

    def detect(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Any]:
        """
        检测图像中的ChArUco标记
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 检测ArUco标记
        marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters)
            
        if marker_ids is None:
            return None, None, None
            
        # 提取ChArUco角点
        success, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.board)
            
        if success:
            return charuco_corners, charuco_ids, rejected
        return None, None, None

    def estimate_pose(self, corners: np.ndarray, ids: np.ndarray,
                     camera_matrix: np.ndarray, 
                     dist_coeffs: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        估计ChArUco板的位姿
        """
        if corners is None or ids is None:
            return None, None, None
            
        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            corners, ids, self.board, camera_matrix, dist_coeffs, None, None)
            
        if success:
            # 使用RMSE计算重投影误差
            obj_points = self.board.getChessboardCorners()
            obj_points = obj_points[ids]
            
            self.rep_error = np.sqrt(np.mean((corners.reshape(-1, 2) - 
                cv2.projectPoints(obj_points, rvec, tvec, 
                camera_matrix, dist_coeffs)[0].reshape(-1, 2)) ** 2))
                
            return np.array([rvec]), np.array([tvec]), self.rep_error
            
        return None, None, None

    def draw_results(self, image: np.ndarray, corners: np.ndarray, ids: np.ndarray,
                    rvecs: Optional[np.ndarray] = None, tvecs: Optional[np.ndarray] = None,
                    camera_matrix: Optional[np.ndarray] = None, 
                    dist_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        绘制检测和位姿估计结果
        """
        if corners is None or ids is None:
            return image
            
        img = image.copy()
        # 绘制ChArUco角点
        cv2.aruco.drawDetectedCornersCharuco(img, corners, ids)
        
        if rvecs is not None and camera_matrix is not None:
            # 绘制坐标轴
            cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, 
                            rvecs[0], tvecs[0], self.config.square_size)
            
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

    def generate_board(self, size: Tuple[int, int] = (1000, 1000)) -> np.ndarray:
        """生成ChArUco标定板图像"""
        board_image = self.board.draw(size)
        return board_image