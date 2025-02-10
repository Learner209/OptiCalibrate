import cv2
import numpy as np
from .base_marker import BaseMarker
from typing import Optional, Tuple, Any
from dataclasses import dataclass
from typing import List, Dict
from loguru import logger


@dataclass
class ChArUcoConfig:
    dictionary: str
    board_size: List[int]
    square_size: float
    marker_size: float
    parameters: Dict[str, float]


class ChArUco(BaseMarker):
    def __init__(self, **kwargs):
        # 创建配置对象并传递给父类
        config = ChArUcoConfig(**kwargs)
        super().__init__(config)

        try:
            # 获取预定义的字典
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(
                getattr(cv2.aruco, self.config.dictionary),
            )

            # 创建ChArUco板
            self.board = cv2.aruco.CharucoBoard(
                tuple(self.config.board_size),
                self.config.square_size,
                self.config.marker_size,
                self.aruco_dict,
            )

            # 创建检测器参数
            self.parameters = cv2.aruco.DetectorParameters()

            # 设置检测器参数
            for param_name, param_value in self.config.parameters.items():
                if hasattr(self.parameters, param_name):
                    setattr(self.parameters, param_name, param_value)
                else:
                    logger.warning(f"未知的参数: {param_name}")

            self.rep_error = None

            logger.info("ChArUco标记检测器初始化完成")

        except Exception as e:
            logger.error(f"ChArUco初始化失败: {str(e)}")
            raise

    def detect(
        self,
        image: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Any]:
        """
        检测图像中的ChArUco标记
        Args:
            image: 输入图像
        Returns:
            corners: 角点坐标
            ids: 标记ID
            rejected: 被拒绝的候选区域
        """
        try:
            gray = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(image.shape) == 3
                else image
            )

            # 检测ArUco标记
            marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.parameters,
            )

            if marker_ids is None:
                return None, None, None

            # 提取ChArUco角点
            success, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                self.board,
            )

            if success:
                return charuco_corners, charuco_ids, rejected
            return None, None, None

        except Exception as e:
            logger.error(f"ChArUco检测失败: {str(e)}")
            return None, None, None

    def estimate_pose(
        self,
        corners: np.ndarray,
        ids: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        估计ChArUco标记的位姿
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
            if corners is None or ids is None:
                return None, None, None

            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                corners,
                ids,
                self.board,
                camera_matrix,
                dist_coeffs,
                None,
                None,
            )

            if retval:
                # 计算重投影误差
                projected_points, _ = cv2.projectPoints(
                    self.board.getChessboardCorners(),
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                )

                # 只计算检测到的角点的误差
                detected_indices = [
                    i for i, id_ in enumerate(self.board.getIds()) if id_ in ids
                ]
                if detected_indices:
                    self.rep_error = np.sqrt(
                        np.mean(
                            (
                                corners.reshape(-1, 2)
                                - projected_points[detected_indices].reshape(-1, 2)
                            )
                            ** 2,
                        ),
                    )
                    return np.array([rvec]), np.array([tvec]), self.rep_error

            return None, None, None

        except Exception as e:
            logger.error(f"ChArUco位姿估计失败: {str(e)}")
            return None, None, None

    def draw_results(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        ids: np.ndarray,
        rvecs: Optional[np.ndarray] = None,
        tvecs: Optional[np.ndarray] = None,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
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
            if corners is None or ids is None:
                return image

            # 绘制检测到的角点
            img = cv2.aruco.drawDetectedCornersCharuco(image.copy(), corners, ids)

            # 如果有位姿信息，绘制坐标轴
            if rvecs is not None and tvecs is not None and camera_matrix is not None:
                cv2.drawFrameAxes(
                    img,
                    camera_matrix,
                    dist_coeffs,
                    rvecs[0],
                    tvecs[0],
                    self.config.square_size,
                )

                # 根据重投影误差显示
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

                    cv2.putText(
                        img,
                        f"RMSE: {self.rep_error:.4f}px",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2,
                    )
            return img

        except Exception as e:
            logger.error(f"ChArUco结果绘制失败: {str(e)}")
            return image

    def generate_board(self, size: Tuple[int, int] = (1000, 1000)) -> np.ndarray:
        """生成ChArUco标定板图像"""
        board_image = self.board.draw(size)
        return board_image
