import cv2
import numpy as np
from .base_marker import BaseMarker
from typing import Optional, Tuple, Any, Dict
from dataclasses import dataclass
from typing import List


@dataclass
class CircleGridConfig:
    pattern_size: List[int]
    circle_size: float
    spacing: float
    symmetric_grid: bool
    blob_params: Dict[str, Any]


class CircleGrid(BaseMarker):
    def __init__(self, **kwargs):
        # 将kwargs转换为配置对象
        self.config = CircleGridConfig(**kwargs)
        self.pattern_size = tuple(self.config.pattern_size)  # (rows, cols)
        self.circle_size = self.config.circle_size  # 圆点直径(米)
        self.spacing = self.config.spacing  # 圆点间距(米)
        self.is_symmetric = self.config.symmetric_grid  # 是否为对称网格

        # 设置圆点检测参数
        self.blob_params = cv2.SimpleBlobDetector_Params()

        # 配置圆点检测器参数
        params = self.config.blob_params
        self.blob_params.minThreshold = params.get("minThreshold", 10)
        self.blob_params.maxThreshold = params.get("maxThreshold", 220)
        self.blob_params.thresholdStep = params.get("thresholdStep", 10)

        self.blob_params.filterByArea = params.get("filterByArea", True)
        self.blob_params.minArea = params.get("minArea", 50)
        self.blob_params.maxArea = params.get("maxArea", 5000)

        self.blob_params.filterByCircularity = params.get("filterByCircularity", True)
        self.blob_params.minCircularity = params.get("minCircularity", 0.8)

        self.blob_params.filterByConvexity = params.get("filterByConvexity", True)
        self.blob_params.minConvexity = params.get("minConvexity", 0.87)

        self.blob_params.filterByInertia = params.get("filterByInertia", True)
        self.blob_params.minInertiaRatio = params.get("minInertiaRatio", 0.1)

        # 创建圆点检测器
        self.detector = cv2.SimpleBlobDetector_create(self.blob_params)

        # 计算3D点坐标
        self.obj_points = self._create_object_points()
        self.rep_error = None

    def _create_object_points(self) -> np.ndarray:
        """
        创建标准3D点坐标
        """
        rows, cols = self.pattern_size
        obj_points = np.zeros((rows * cols, 3), np.float32)
        for i in range(rows):
            for j in range(cols):
                obj_points[i * cols + j] = [j * self.spacing, i * self.spacing, 0]
        return obj_points

    def detect(
        self,
        image: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Any]:
        """
        检测圆点格
        Args:
            image: 输入图像
        Returns:
            corners: 检测到的圆点中心坐标
            pattern_was_found: 是否检测到完整的圆点格
            keypoints: 检测到的所有圆点
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 增强对比度
        gray = cv2.equalizeHist(gray)

        # 检测圆点
        if self.is_symmetric:
            pattern_was_found, corners = cv2.findCirclesGrid(
                gray,
                self.pattern_size,
                flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                blobDetector=self.detector,
            )
        else:
            pattern_was_found, corners = cv2.findCirclesGrid(
                gray,
                self.pattern_size,
                flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                blobDetector=self.detector,
            )

        # 获取所有检测到的圆点
        keypoints = self.detector.detect(gray)

        if pattern_was_found:
            # 亚像素精确化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            return corners, np.ones((len(corners), 1)), keypoints

        return None, None, keypoints

    def estimate_pose(
        self,
        corners: np.ndarray,
        ids: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        估计圆点格的位姿
        """
        if corners is None:
            return None, None, None

        # 求解PnP
        ret, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if ret:
            # 计算重投影误差
            proj_points, _ = cv2.projectPoints(
                self.obj_points,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
            self.rep_error = np.sqrt(
                np.mean((corners.reshape(-1, 2) - proj_points.reshape(-1, 2)) ** 2),
            )
            return np.array([rvec]), np.array([tvec]), self.rep_error

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
        """
        img = image.copy()

        if corners is not None:
            # 绘制检测到的圆点
            cv2.drawChessboardCorners(img, self.pattern_size, corners, True)

            # 绘制圆点序号
            for i, corner in enumerate(corners):
                cv2.putText(
                    img,
                    str(i),
                    tuple(corner.ravel().astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            if rvecs is not None and camera_matrix is not None:
                # 绘制坐标轴
                cv2.drawFrameAxes(
                    img,
                    camera_matrix,
                    dist_coeffs,
                    rvecs[0],
                    tvecs[0],
                    self.spacing * 2,
                )

                if self.rep_error is not None:
                    # 根据误差大小选择颜色
                    if self.rep_error < 0.5:
                        color = (0, 255, 0)  # 绿色：优秀
                    elif self.rep_error < 1.0:
                        color = (0, 255, 255)  # 黄色：良好
                    elif self.rep_error < 1.5:
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

    def generate_pattern(self, size: Tuple[int, int] = (1000, 1000)) -> np.ndarray:
        """
        生成圆点格标定板图像
        """
        img = np.ones(size, dtype=np.uint8) * 255
        margin = 50  # 边距

        # 计算圆点实际大小
        total_width = size[1] - 2 * margin
        total_height = size[0] - 2 * margin

        spacing_pixels = min(
            total_width / (self.pattern_size[1] - 1),
            total_height / (self.pattern_size[0] - 1),
        )
        circle_radius = int(spacing_pixels * self.circle_size / self.spacing / 2)

        # 绘制圆点
        for i in range(self.pattern_size[0]):
            for j in range(self.pattern_size[1]):
                if not self.is_symmetric or (i % 2 == 0):
                    x = int(margin + j * spacing_pixels)
                else:
                    x = int(margin + (j + 0.5) * spacing_pixels)
                y = int(margin + i * spacing_pixels)

                cv2.circle(img, (x, y), circle_radius, 0, -1)

        return img
