import cv2
import numpy as np
import asyncio
from loguru import logger
from typing import List, Tuple


class CalibrationDataCollector:
    def __init__(self, robot, camera, marker_detector):
        """
        初始化数据采集器

        Args:
            robot: 机器人对象
            camera: 相机对象
            marker_detector: 标记检测器对象
        """
        self.robot = robot
        self.camera = camera
        self.marker_detector = marker_detector
        self.robot_poses: List[np.ndarray] = []
        self.marker_poses: List[Tuple[np.ndarray, np.ndarray]] = []
        self.running = False

        logger.info("数据采集器初始化完成")

    async def collect_data(
        self,
    ) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        """
        异步采集标定数据
        """
        logger.info("开始采集数据")
        logger.info("按 's' 键保存当前姿态，按 'ESC' 键结束采集")

        self.running = True

        while self.running:
            # 获取机器人位姿
            robot_pose = await asyncio.get_event_loop().run_in_executor(
                None,
                self.robot.get_current_pose,
            )
            if robot_pose is None:
                logger.warning("无法获取机器人位姿")
                continue

            # 捕获图像
            color_frame, _ = await asyncio.get_event_loop().run_in_executor(
                None,
                self.camera.capture_frame,
            )
            if color_frame is None:
                logger.warning("无法获取图像")
                continue

            # 检测标记
            corners, ids, rejected = await asyncio.get_event_loop().run_in_executor(
                None,
                self.marker_detector.detect,
                color_frame,
            )
            if corners is not None and len(corners) > 0:
                # 估计标记位姿
                (
                    rvecs,
                    tvecs,
                    rep_error,
                ) = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.marker_detector.estimate_pose,
                    corners,
                    ids,
                    self.camera.get_intrinsics(),
                    self.camera.get_distortion(),
                )

                if rvecs is not None:
                    # 绘制结果
                    result_img = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.marker_detector.draw_results,
                        color_frame.copy(),
                        corners,
                        ids,
                        rvecs,
                        tvecs,
                        self.camera.get_intrinsics(),
                        self.camera.get_distortion(),
                    )

                    # 显示采样计数
                    cv2.putText(
                        result_img,
                        f"Samples: {len(self.robot_poses)}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    # 显示图像
                    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
                    cv2.imshow("Calibration", result_img)

                    # 处理按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("s"):  # 's' 键保存当前姿态
                        self.robot_poses.append(robot_pose)
                        self.marker_poses.append((rvecs[0], tvecs[0]))
                        logger.info(f"保存第 {len(self.robot_poses)} 组数据")
                    elif key == 27:  # ESC 键结束采集
                        self.running = False
            else:
                cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
                cv2.imshow("Calibration", color_frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    self.running = False

            # 添加短暂延时，让出CPU时间
            await asyncio.sleep(0.01)

        logger.info(f"数据采集完成，共采集 {len(self.robot_poses)} 组数据")
        return self.robot_poses, self.marker_poses
