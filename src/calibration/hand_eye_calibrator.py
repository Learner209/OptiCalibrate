import cv2
import numpy as np
import transforms3d as tfs
from loguru import logger
import json
from typing import List, Optional, Tuple
from pathlib import Path

class HandEyeCalibrator:
    def __init__(self, 
                 method: str = "eye_in_hand",
                 calib_method: str = "CALIB_HAND_EYE_TSAI",
                 output_path: str = "calibrations/result.json",
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        初始化手眼标定器
        
        Args:
            method: 标定方法，可选 "eye_in_hand" 或 "eye_to_hand"
            calib_method: OpenCV标定方法，例如 "CALIB_HAND_EYE_TSAI"
            output_path: 结果保存路径
            camera_matrix: 相机内参矩阵
            dist_coeffs: 相机畸变系数
        """
        self.method = method
        self.calib_method = getattr(cv2, calib_method)  # 转换为 OpenCV 常量
        self.output_path = output_path
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # 创建输出目录
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"手眼标定器初始化完成，使用{method}方法")

    def calibrate(self, robot_poses: List[np.ndarray], 
                 marker_poses: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        执行手眼标定
        
        Args:
            robot_poses: 机器人位姿列表，每个位姿为 [qx, qy, qz, qw, x, y, z]
            marker_poses: 标记位姿列表，每个位姿为 (rvec, tvec)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 旋转矩阵R和平移向量T,失败则返回 (None, None)
        """
        logger.info(f"开始{self.method}标定，使用{len(robot_poses)}组样本")
        
        if len(robot_poses) < 2 or len(marker_poses) < 2:
            logger.error("样本数量不足,至少需要2组样本")
            return None, None

        if len(robot_poses) != len(marker_poses):
            logger.error("机器人位姿和标记位姿数量不匹配")
            return None, None

        if self.method == "eye_in_hand":
            return self.calibrate_eye_in_hand(robot_poses, marker_poses)
        elif self.method == "eye_to_hand":
            return self.calibrate_eye_to_hand(robot_poses, marker_poses)
        else:
            logger.error(f"未知的标定方法: {self.method}")
            return None, None

    def calibrate_eye_in_hand(self, robot_poses: List[np.ndarray], 
                            marker_poses: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """手眼标定：相机安装在机器人末端"""
        R_gripper2base, T_gripper2base = [], []
        R_target2cam, T_target2cam = [], []

        for robot_pose, marker_pose in zip(robot_poses, marker_poses):
            # 处理机器人末端到基座的变换
            quat = robot_pose[:4]
            trans = robot_pose[4:]
            r = tfs.quaternions.quat2mat(quat)
            T_gripper2base.append(trans)
            R_gripper2base.append(r)

            # 处理标定板到相机的变换
            rvec, tvec = marker_pose
            r, _ = cv2.Rodrigues(rvec)
            T_target2cam.append(tvec.flatten())
            R_target2cam.append(r)

        return self._perform_calibration(R_gripper2base, T_gripper2base, R_target2cam, T_target2cam)

    def calibrate_eye_to_hand(self, robot_poses: List[np.ndarray], 
                            marker_poses: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """手眼标定：相机固定在工作空间"""
        R_base2gripper, T_base2gripper = [], []
        R_target2cam, T_target2cam = [], []

        for robot_pose, marker_pose in zip(robot_poses, marker_poses):
            # 处理基座到机器人末端的变换
            quat = robot_pose[:4]
            trans = robot_pose[4:]
            r = tfs.quaternions.quat2mat(quat)
            r = np.linalg.inv(r)
            t = -np.dot(r, trans)
            T_base2gripper.append(t)
            R_base2gripper.append(r)

            # 处理标定板到相机的变换
            rvec, tvec = marker_pose
            r, _ = cv2.Rodrigues(rvec)
            T_target2cam.append(tvec.flatten())
            R_target2cam.append(r)

        return self._perform_calibration(R_base2gripper, T_base2gripper, R_target2cam, T_target2cam)

    def _perform_calibration(self, R1: List[np.ndarray], T1: List[np.ndarray], 
                           R2: List[np.ndarray], T2: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """执行标定计算"""
        # 转换为NumPy数组
        R1 = np.array(R1, dtype=np.float64)
        T1 = np.array(T1, dtype=np.float64)
        R2 = np.array(R2, dtype=np.float64)
        T2 = np.array(T2, dtype=np.float64)

        try:
            R, T = cv2.calibrateHandEye(R1, T1, R2, T2, method=self.calib_method)
            if R is None or T is None:
                logger.error("标定失败：返回空值")
                return None, None
            
            logger.info("标定完成")
            return R, T
        except cv2.error as e:
            logger.error(f"OpenCV标定错误: {str(e)}")
            return None, None

    def save_result(self, R: Optional[np.ndarray], T: Optional[np.ndarray]) -> None:
        """保存标定结果"""
        if R is None or T is None:
            logger.error("无法保存结果:R或T为空")
            return
        
        T_result = np.eye(4)
        T_result[:3, :3] = R
        T_result[:3, 3] = T.flatten()

        # 将numpy数组转换为列表
        result_list = T_result.tolist()

        # 保存为JSON文件
        try:
            with open(self.output_path, 'w') as f:
                json.dump(result_list, f, indent=2)
            logger.info(f"标定结果已保存至 {self.output_path}")
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")