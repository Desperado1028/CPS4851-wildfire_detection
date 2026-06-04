"""
MobileNetV3-Small 火灾分类器
适用于边缘设备，参数量约 2.5M
"""
import torch
import torch.nn as nn
import torchvision.models as models

class WildfireClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        # 使用 MobileNetV3-Small 作为骨干
        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # 替换分类头
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def export_to_onnx(self, onnx_path: str, input_size=(1, 3, 224, 224)):
        """导出为 ONNX 格式用于边缘推理"""
        self.eval()
        dummy_input = torch.randn(*input_size)
        torch.onnx.export(
            self, dummy_input, onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11
        )
        print(f"模型已导出为 ONNX: {onnx_path}")