import torch
import sys
sys.path.append('.')
from model.wildfire_classifier import WildfireClassifier

# 加载训练好的 .pth
model = WildfireClassifier(num_classes=2, pretrained=False)
model.load_state_dict(torch.load('models/wildfire_mobilenetv3.pth', map_location='cpu'))
model.eval()

# 导出 ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    'models/wildfire_mobilenetv3.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=11
)
print("ONNX 导出成功！")