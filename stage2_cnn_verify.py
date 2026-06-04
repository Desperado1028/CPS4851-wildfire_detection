import cv2
import numpy as np
import os
from typing import Tuple, Optional, List
from config import Stage2Config

class CNNVerifier:
    """
    阶段2：对阶段1的候选区域进行 CNN 分类验证
    支持两种后端：
      - ONNX Runtime（推荐，边缘优化）
      - PyTorch（备用）
    仅在阶段1触发时运行，节省计算资源
    """
    def __init__(self, config: Stage2Config):
        self.config = config
        self.model = None
        self.backend = None  # 'onnx' or 'pytorch' or 'fallback'

        # 尝试加载模型
        if config.use_onnx and os.path.exists(config.model_path):
            self._load_onnx()
        elif os.path.exists(config.torch_model_path):
            self._load_pytorch()
        else:
            print("[Stage2] 未找到训练好的模型，使用基于颜色特征的备用分类器")
            self.backend = 'fallback'

    def _load_onnx(self):
        """加载 ONNX 模型"""
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                self.config.model_path,
                providers=['CPUExecutionProvider']
            )
            self.backend = 'onnx'
            print(f"[Stage2] ONNX 模型已加载: {self.config.model_path}")
        except Exception as e:
            print(f"[Stage2] ONNX 加载失败: {e}")
            self.backend = None

    def _load_pytorch(self):
        """加载 PyTorch 模型"""
        try:
            import torch
            from model.wildfire_classifier import WildfireClassifier
            self.torch_model = WildfireClassifier(num_classes=2)
            self.torch_model.load_state_dict(
                torch.load(self.config.torch_model_path, map_location='cpu')
            )
            self.torch_model.eval()
            self.backend = 'pytorch'
            print(f"[Stage2] PyTorch 模型已加载: {self.config.torch_model_path}")
        except Exception as e:
            print(f"[Stage2] PyTorch 加载失败: {e}")
            self.backend = None

    def preprocess_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """裁剪并预处理 ROI 为模型输入格式"""
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        # 调整大小
        roi = cv2.resize(roi, self.config.input_size)
        # BGR -> RGB
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # 归一化 [0, 1]
        roi = roi.astype(np.float32) / 255.0
        # 标准化 (ImageNet 统计值)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        roi = (roi - mean) / std
        # HWC -> CHW
        roi = np.transpose(roi, (2, 0, 1))
        return roi

    def _onnx_inference(self, input_tensor: np.ndarray) -> Tuple[int, float]:
        """ONNX 推理"""
        input_batch = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        outputs = self.session.run(None, {'input': input_batch})
        logits = outputs[0][0]
        probs = self._softmax(logits)
        class_id = np.argmax(probs)
        confidence = probs[class_id]
        return class_id, confidence

    def _pytorch_inference(self, input_tensor: np.ndarray) -> Tuple[int, float]:
        """PyTorch 推理"""
        import torch
        with torch.no_grad():
            input_batch = torch.from_numpy(input_tensor).unsqueeze(0)
            logits = self.torch_model(input_batch)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        return class_id, confidence

    def _fallback_inference(self, roi: np.ndarray) -> Tuple[int, float]:
        """
        备用分类器：基于颜色直方图特征
        分析火焰颜色的占比来估计火灾概率
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 火焰颜色范围
        lower_fire = np.array([0, 80, 80])
        upper_fire = np.array([35, 255, 255])
        fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
        fire_ratio = np.sum(fire_mask > 0) / fire_mask.size

        # 亮度分析（烟雾通常降低对比度）
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)

        # 综合评分
        score = fire_ratio * 0.7 + max(0, (1 - std_dev / 128)) * 0.3
        score = np.clip(score, 0, 1)

        if score > 0.3:
            return 1, score  # fire_smoke
        else:
            return 0, 1 - score  # normal

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def verify(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, float, str]:
        """
        对单个候选区域进行分类
        返回：(类别ID, 置信度, 类别名称)
        """
        roi_raw = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        if roi_raw.size == 0:
            return 0, 0.0, "normal"

        try:
            if self.backend == 'onnx':
                input_tensor = self.preprocess_roi(frame, bbox)
                if input_tensor is None:
                    return 0, 0.0, "normal"
                class_id, confidence = self._onnx_inference(input_tensor)
            elif self.backend == 'pytorch':
                input_tensor = self.preprocess_roi(frame, bbox)
                if input_tensor is None:
                    return 0, 0.0, "normal"
                class_id, confidence = self._pytorch_inference(input_tensor)
            else:
                class_id, confidence = self._fallback_inference(roi_raw)

            class_name = self.config.class_names[class_id] if class_id < len(self.config.class_names) else "unknown"
            return class_id, confidence, class_name

        except Exception as e:
            print(f"[Stage2] 推理异常: {e}")
            return 0, 0.0, "normal"

    def verify_batch(self, frame: np.ndarray, candidates: List[Tuple[int, int, int, int]]) -> List[dict]:
        """批量验证候选区域"""
        results = []
        for bbox in candidates:
            class_id, conf, name = self.verify(frame, bbox)
            results.append({
                'bbox': bbox,
                'class_id': class_id,
                'confidence': conf,
                'class_name': name,
                'is_fire': class_id == 1 and conf >= self.config.confidence_threshold
            })
        return results