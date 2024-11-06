import cv2
import numpy as np
from .base_marker import BaseMarker
from omegaconf import DictConfig
from typing import Optional, Tuple, Any, Dict
from dataclasses import dataclass
from typing import List

@dataclass
class ChessboardConfig:
    pattern_size: List[int]
    square_size: float
    parameters: Dict[str, Any]

class Chessboard(BaseMarker):
    def __init__(self, **kwargs):
        # 将kwargs转换为配置对象
        self.config = ChessboardConfig(**kwargs)
        self.pattern_size = tuple(self.config.pattern_size)
        self.square_size = self.config.square_size
        
        # 设置棋盘格检测参数
        self.parameters = self.config.parameters
        
        # 预计算对象点
        self.objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.pattern_size[0], 
                                   0:self.pattern_size[1]].T.reshape(-1,2) * self.square_size
        
        self.rep_error = None

    def detect(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Any]:
        """
        检测图像中的棋盘格角点
        Args:
            image: 输入图像
        Returns:
            corners: 角点坐标
            ids: 标记ID(棋盘格没有ID,这里返回None)
            rejected: 被拒绝的候选区域(这里返回None)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        found, corners = cv2.findChessboardCorners(
            gray, self.pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                  cv2.CALIB_CB_FILTER_QUADS
        )
        
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            return corners, None, None
        return None, None, None

    def estimate_pose(self, corners: np.ndarray, ids: np.ndarray,
                 camera_matrix: np.ndarray, 
                 dist_coeffs: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        估计棋盘格的位姿
        """
        if corners is None:
            return None, None, None
                
        retval, rvec, tvec = cv2.solvePnP(
            self.objp, corners, camera_matrix, dist_coeffs)
        
        if retval:
            # 使用RMSE计算重投影误差
            self.rep_error = np.sqrt(np.mean((corners.reshape(-1, 2) - 
                cv2.projectPoints(self.objp, rvec, tvec, 
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
        if corners is None:
            return image
        
        img = image.copy()
        cv2.drawChessboardCorners(img, self.pattern_size, corners, True)
        
        if rvecs is not None and camera_matrix is not None:
            cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, 
                            rvecs[0], tvecs[0], self.square_size * 2)
            
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