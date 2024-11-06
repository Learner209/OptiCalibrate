from abc import ABC, abstractmethod
from omegaconf import DictConfig

class BaseRobot(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def get_current_pose(self):
        """
        获取机器人当前位姿
        
        Returns:
            list/numpy.ndarray: 机器人位姿 [x, y, z, rx, ry, rz] 或 None(如果发生错误)
        """
        pass 