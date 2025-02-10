import sys
sys.path.append(".")
import flexivrdk
from .base_robot import BaseRobot
from loguru import logger
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class FlexivConfig:
    robot_ip: str
    local_ip: str

class Flexiv(BaseRobot):
    def __init__(self, **kwargs):
        # 创建配置对象并传递给父类
        config = FlexivConfig(**kwargs)
        super().__init__(config)
        
        try:
            # 初始化机器人连接
            logger.info(f"正在连接Flexiv机器人 {self.config.robot_ip}...")
            self._robot = flexivrdk.Robot(self.config.robot_ip, self.config.local_ip)
            
            # 使能机器人
            logger.info("正在使能机器人...")
            self._robot.enable()
            logger.info("机器人初始化完成")
            
        except Exception as e:
            logger.error(f"机器人初始化失败: {str(e)}")
            raise

    def get_current_pose(self) -> Optional[List[float]]:
        """
        获取机器人当前位姿
        
        Returns:
            List[float]: 机器人位姿 [qx, qy, qz, qw, x, y, z] 或 None(如果发生错误)
        """
        try:
            robot_states = flexivrdk.RobotStates()
            self._robot.getRobotStates(robot_states)
            return robot_states.tcpPose
        except Exception as e:
            logger.error(f"获取机器人状态失败: {str(e)}")
            return None

    def __del__(self):
        """
        析构函数，确保机器人正确关闭
        """
        if hasattr(self, '_robot'):
            try:
                logger.info("正在关闭机器人连接...")
                self._robot.stop()
            except Exception as e:
                logger.error(f"关闭机器人失败: {str(e)}") 