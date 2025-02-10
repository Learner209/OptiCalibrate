import cv2
import numpy as np
from .base_marker import BaseMarker
from omegaconf import DictConfig
from typing import Optional, Tuple, Any, Dict
from dataclasses import dataclass
from typing import List
from loguru import logger

@dataclass
class ChessboardConfig:
    pattern_size: List[int]
    square_size: float
    parameters: Dict[str, Any]

class Chessboard(BaseMarker):
    def __init__(self, **kwargs):
        # 创建配置对象并传递给父类
        config = ChessboardConfig(**kwargs)
        super().__init__(config)
        
        try:
            self.pattern_size = tuple(self.config.pattern_size)
            self.square_size = self.config.square_size
            
            # 设置棋盘格检测参数
            self.parameters = self.config.parameters
            
            # 预计算对象点
            self.objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
            self.objp[:,:2] = np.mgrid[0:self.pattern_size[0], 
                                      0:self.pattern_size[1]].T.reshape(-1,2) * self.square_size
            
            self.rep_error = None
            
            logger.info("棋盘格检测器初始化完成")
            
        except Exception as e:
            logger.error(f"棋盘格初始化失败: {str(e)}")
            raise

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
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            # import ipdb; ipdb.set_trace()
            found, corners = cv2.findChessboardCorners(
                gray, self.pattern_size,
                flags=sum(eval(flag) for flag in self.parameters['flags'])  # 将标志位列表转换为标志位
            )
            
            if found:
                criteria = (
                    sum(eval(type) for type in self.parameters['criteria']['type']),  # 将类型列表转换为标志位
                    self.parameters['criteria']['maxCount'],
                    self.parameters['criteria']['epsilon']
                )
                corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                return corners, None, None
            return None, None, None
            
        except Exception as e:
            logger.error(f"棋盘格检测失败: {str(e)}")
            return None, None, None

    def estimate_pose(self, corners: np.ndarray, ids: np.ndarray,
                     camera_matrix: np.ndarray, 
                     dist_coeffs: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        估计棋盘格的位姿
        Args:
            corners: 角点坐标
            ids: 标记ID(对棋盘格无用)
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
        Returns:
            rvecs: 旋转向量
            tvecs: 平移向量
            rep_error: 重投影误差
        """
        try:
            if corners is None:
                return None, None, None

            # import ipdb; ipdb.set_trace()
            retval, rvec, tvec = cv2.solvePnP(
                self.objp, corners, camera_matrix, dist_coeffs)
            
            if retval:
                # 计算重投影误差
                projected_points, _ = cv2.projectPoints(
                    self.objp, rvec, tvec, camera_matrix, dist_coeffs)
                self.rep_error = np.sqrt(np.mean(
                    (corners.reshape(-1, 2) - projected_points.reshape(-1, 2)) ** 2))
                return np.array([rvec]), np.array([tvec]), self.rep_error
            
            return None, None, None
            
        except Exception as e:
            logger.error(f"棋盘格位姿估计失败: {str(e)}")
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
            ids: 标记ID(对棋盘格无用)
            rvecs: 旋转向量
            tvecs: 平移向量
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
        Returns:
            绘制结果图像
        """
        try:
            if corners is None:
                return image
            
            img = cv2.drawChessboardCorners(image.copy(), self.pattern_size, corners, True)
            
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
            
        except Exception as e:
            logger.error(f"棋盘格结果绘制失败: {str(e)}")
            return image