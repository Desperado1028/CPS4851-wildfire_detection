from collections import deque
import numpy as np
from typing import List, Optional
from config import Stage3Config

class TemporalDecisionEngine:
    """
    阶段3：时序决策
    - 维护滑动窗口内的检测历史
    - 使用指数加权移动平均（EMA）平滑
    - 当窗口内正例比例超过阈值时触发警报
    - 包含警报冷却机制，避免重复触发
    """
    def __init__(self, config: Stage3Config):
        self.config = config
        self.window: deque = deque(maxlen=config.window_size)
        self.ema_score: float = 0.0
        self.alert_active: bool = False
        self.frames_since_alert: int = 0
        self.alert_count: int = 0

    def update(self, stage2_results: List[dict]) -> dict:
        """
        输入阶段2的验证结果，输出当前帧的决策
        """
        # 计算当前帧的火灾分数
        if len(stage2_results) == 0:
            frame_score = 0.0
        else:
            # 取所有候选区域中最高置信度（仅考虑火灾类）
            fire_confs = [r['confidence'] for r in stage2_results if r['class_id'] == 1]
            frame_score = max(fire_confs) if fire_confs else 0.0

        # 加入滑动窗口
        self.window.append(frame_score)

        # EMA 平滑
        alpha = self.config.smoothing_alpha
        self.ema_score = alpha * frame_score + (1 - alpha) * self.ema_score

        # 窗口统计
        window_array = np.array(list(self.window))
        positive_count = np.sum(window_array >= 0.5)  # 置信度>=0.5视为正例
        positive_ratio = positive_count / len(self.window) if len(self.window) > 0 else 0

        # 决策逻辑
        trigger_alert = False
        alert_message = None

        self.frames_since_alert += 1

        # 触发条件：正例比例超过阈值 AND 不在冷却期
        if (positive_ratio >= self.config.alert_threshold and
            self.frames_since_alert >= self.config.cooldown_frames):
            trigger_alert = True
            alert_message = f"⚠ FIRE ALERT! Positive ratio: {positive_ratio:.1%}"
            self.frames_since_alert = 0
            self.alert_count += 1

        # 解除警报条件：EMA分数持续低
        if self.alert_active and self.ema_score < 0.2 and positive_ratio < 0.1:
            self.alert_active = False

        if trigger_alert:
            self.alert_active = True

        return {
            'frame_score': frame_score,
            'ema_score': self.ema_score,
            'positive_ratio': positive_ratio,
            'window_size': len(self.window),
            'alert_triggered': trigger_alert,
            'alert_active': self.alert_active,
            'alert_message': alert_message,
            'total_alerts': self.alert_count,
        }

    def reset(self):
        """重置状态"""
        self.window.clear()
        self.ema_score = 0.0
        self.alert_active = False
        self.frames_since_alert = 0