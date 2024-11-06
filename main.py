import cv2
import hydra
from omegaconf import DictConfig
from loguru import logger
from hydra.utils import instantiate

@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig):
    try:
        # 初始化机器人
        logger.info("正在初始化机器人...")
        robot = instantiate(cfg.robot)
        if robot is None:
            logger.error("机器人初始化失败")
            return

        # 初始化相机
        logger.info("正在初始化相机...")
        camera = instantiate(cfg.camera)
        if camera is None:
            logger.error("相机初始化失败")
            return

        # 初始化标记检测器
        logger.info("正在初始化标记检测器...")
        marker_detector = instantiate(cfg.marker)
        if marker_detector is None:
            logger.error("标记检测器初始化失败")
            return
            
        # 初始化手眼标定器
        logger.info("正在初始化手眼标定器...")
        calibrator = instantiate(cfg.calibration)
        if calibrator is None:
            logger.error("手眼标定器初始化失败")
            return

        # 初始化数据采集器
        collector = instantiate(
            cfg.collector,
            robot=robot,
            camera=camera,
            marker_detector=marker_detector
        )

        logger.info("系统初始化完成")

        # 采集标定数据
        robot_poses, marker_poses = collector.collect_data()
        
        if len(robot_poses) < 2:
            logger.error("采集的数据不足,至少需要2组数据")
            return
            
        # 执行手眼标定
        R, T = calibrator.calibrate(robot_poses, marker_poses)
        
        if R is not None and T is not None:
            # 保存标定结果 
            calibrator.save_result(R, T)
            
            # 显示标定结果
            logger.info("\n标定结果:")
            logger.info(f"旋转矩阵 R:\n{R}")
            logger.info(f"平移向量 T:\n{T.flatten()}")
        else:
            logger.error("标定失败")

    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        logger.error("", exc_info=True)
    finally:
        cv2.destroyAllWindows()
        logger.info("程序已终止")

if __name__ == "__main__":
    # 配置logger
    logger.add("logs/calibration_{time}.log", 
               rotation="1 day", 
               retention="1 week",
               level="DEBUG")
    
    main()