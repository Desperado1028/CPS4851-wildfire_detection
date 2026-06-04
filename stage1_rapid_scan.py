import cv2
import numpy as np
from typing import List, Tuple, Optional
from config import Stage1Config

class RapidScanner:
    """
    阶段1：极低计算成本的快速筛选
    - 使用 HSV 颜色空间检测火焰/烟雾颜色
    - 使用帧差法检测运动
    - 融合两者生成候选区域
    - 处理时间目标：< 5ms/帧 (CPU)
    """
    def __init__(self, config: Stage1Config):
        self.config = config
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.frame_count = 0

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预处理：降噪、颜色空间转换"""
        blurred = cv2.GaussianBlur(frame, self.config.blur_kernel, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        return hsv, gray

    def detect_fire_color(self, hsv: np.ndarray) -> np.ndarray:
        """基于HSV颜色空间的火焰区域掩码"""
        lower = np.array(self.config.fire_hsv_lower, dtype=np.uint8)
        upper = np.array(self.config.fire_hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def detect_smoke_color(self, hsv: np.ndarray) -> np.ndarray:
        """基于HSV颜色空间的烟雾区域掩码"""
        lower = np.array(self.config.smoke_hsv_lower, dtype=np.uint8)
        upper = np.array(self.config.smoke_hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def detect_motion(self, gray: np.ndarray) -> np.ndarray:
        """帧差法运动检测"""
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray)

        # 计算帧差
        diff = cv2.absdiff(gray, self.prev_gray)
        # 阈值化
        _, motion_mask = cv2.threshold(diff, self.config.motion_threshold, 255, cv2.THRESH_BINARY)
        # 膨胀连接相邻区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

        self.prev_gray = gray
        return motion_mask

    def extract_candidates(self, combined_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """从掩码中提取候选边界框"""
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.config.min_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # 稍微扩大边界框以包含上下文
                x = max(0, x - 5)
                y = max(0, y - 5)
                w = min(w + 10, 640)  # 假设帧宽640
                h = min(h + 10, 480)
                candidates.append((x, y, w, h))
        return candidates

    def scan(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray, bool]:
        """
        执行阶段1扫描
        返回：(候选区域列表, 可视化掩码, 是否触发阶段2)
        """
        hsv, gray = self.preprocess(frame)

        # 火焰颜色掩码
        fire_mask = self.detect_fire_color(hsv)
        # 烟雾颜色掩码
        smoke_mask = self.detect_smoke_color(hsv)
        # 运动掩码
        motion_mask = self.detect_motion(gray)

        # 融合：颜色掩码 AND 运动掩码（减少误检）
        fire_triggered = cv2.bitwise_and(fire_mask, motion_mask)
        smoke_triggered = cv2.bitwise_and(smoke_mask, motion_mask)
        combined = cv2.bitwise_or(fire_triggered, smoke_triggered)

        # 提取候选区域
        candidates = self.extract_candidates(combined)
        triggered = len(candidates) > 0

        # 生成可视化掩码（彩色）
        viz_mask = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        # 火焰候选区域标记为红色
        viz_mask[fire_triggered > 0] = (0, 0, 255)
        # 烟雾候选区域标记为灰色
        viz_mask[smoke_triggered > 0] = (128, 128, 128)

        self.frame_count += 1
        return candidates, viz_mask, triggered