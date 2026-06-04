import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Stage1Config:
    fire_hsv_lower: List[int] = field(default_factory=lambda: [0, 100, 100])
    fire_hsv_upper: List[int] = field(default_factory=lambda: [35, 255, 255])
    smoke_hsv_lower: List[int] = field(default_factory=lambda: [0, 0, 180])
    smoke_hsv_upper: List[int] = field(default_factory=lambda: [180, 30, 255])
    motion_threshold: int = 25
    min_contour_area: int = 500
    blur_kernel: Tuple[int, int] = (5, 5)
    morph_kernel: Tuple[int, int] = (3, 3)

@dataclass
class Stage2Config:
    model_path: str = "models/wildfire_mobilenetv3.onnx"
    input_size: Tuple[int, int] = (224, 224)
    confidence_threshold: float = 0.65
    use_onnx: bool = True
    torch_model_path: str = "models/wildfire_mobilenetv3.pth"
    class_names: List[str] = field(default_factory=lambda: ["normal", "fire_smoke"])

@dataclass
class Stage3Config:
    window_size: int = 30
    alert_threshold: float = 0.5
    cooldown_frames: int = 90
    smoothing_alpha: float = 0.2

@dataclass
class SystemConfig:
    input_source: str = "0"          # 摄像头索引或视频路径即可，比如 "test.mp4"默认0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    enable_display: bool = True
    save_video: bool = True
    output_path: str = "output/"
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)

    def __post_init__(self):
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs("models", exist_ok=True)