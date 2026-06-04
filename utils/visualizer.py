import cv2
import numpy as np
import time
from collections import deque
from typing import List, Tuple, Optional

class Visualizer:
    """绘制检测结果、性能指标、警报状态"""
    def __init__(self):
        self.fps_history = deque(maxlen=30)
        self.stage1_times = deque(maxlen=50)
        self.stage2_times = deque(maxlen=50)
        self.stage1_trigger_count = 0
        self.stage2_trigger_count = 0
        self.total_frames = 0

    def draw_stage1_candidates(self, frame: np.ndarray, contours: List[np.ndarray],
                                color: Tuple = (0, 255, 255)) -> np.ndarray:
        """绘制阶段1候选区域"""
        viz = frame.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(viz, (x, y), (x + w, y + h), color, 2)
            cv2.putText(viz, "S1:Candidate", (x, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return viz

    def draw_stage2_result(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                           label: str, confidence: float, color: Tuple = (0, 255, 0)) -> np.ndarray:
        """绘制阶段2验证结果"""
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"S2:{label} ({confidence:.2f})"
        cv2.putText(frame, text, (x, y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

    def draw_alert(self, frame: np.ndarray, message: str = "FIRE ALERT!") -> np.ndarray:
        """绘制警报覆盖层"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        # 红色半透明覆盖
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        # 警报文字
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        cv2.putText(frame, message, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        return frame

    def draw_status_bar(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        """绘制底部状态栏"""
        h, w = frame.shape[:2]
        bar = np.zeros((100, w, 3), dtype=np.uint8)
        bar[:] = (40, 40, 40)

        y_offset = 25
        for key, value in metrics.items():
            text = f"{key}: {value}"
            cv2.putText(bar, text, (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 22

        return np.vstack([frame, bar])

    def update_metrics(self, s1_time: float, s2_time: Optional[float],
                       s1_triggered: bool, s2_triggered: bool):
        self.total_frames += 1
        self.stage1_times.append(s1_time)
        if s2_time is not None:
            self.stage2_times.append(s2_time)
        if s1_triggered:
            self.stage1_trigger_count += 1
        if s2_triggered:
            self.stage2_trigger_count += 1

    def get_metrics(self) -> dict:
        avg_s1 = np.mean(self.stage1_times) * 1000 if self.stage1_times else 0
        avg_s2 = np.mean(self.stage2_times) * 1000 if self.stage2_times else 0
        s1_rate = (self.stage1_trigger_count / max(self.total_frames, 1)) * 100
        s2_rate = (self.stage2_trigger_count / max(self.total_frames, 1)) * 100

        return {
            "Stage1(ms)": f"{avg_s1:.1f}",
            "Stage2(ms)": f"{avg_s2:.1f}",
            "S1 Trigger%": f"{s1_rate:.1f}%",
            "S2 Trigger%": f"{s2_rate:.2f}%",
            "Total Frames": self.total_frames,
        }