#!/usr/bin/env python3
"""
训练阶段2的 CNN 分类器
使用 Kaggle Fire Detection Dataset 或类似数据集
数据集结构：
  data/
    train/
      fire_smoke/    # 火灾/烟雾图像
      normal/        # 正常森林/户外图像
    val/
      fire_smoke/
      normal/
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import sys
sys.path.append('.')
from model.wildfire_classifier import WildfireClassifier

def train():
    # 超参数
    batch_size = 32
    epochs = 30
    learning_rate = 1e-4
    data_dir = "data"
    model_save_path = "models/wildfire_mobilenetv3.pth"
    onnx_save_path = "models/wildfire_mobilenetv3.onnx"

    # 数据增强与预处理
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"), transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"训练样本: {len(train_dataset)}, 类别: {train_dataset.classes}")
    print(f"验证样本: {len(val_dataset)}")

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WildfireClassifier(num_classes=len(train_dataset.classes), pretrained=True)
    model = model.to(device)
    print(f"使用设备: {device}")

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练循环
    best_acc = 0.0
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  → 模型已保存 (最佳准确率: {best_acc:.2f}%)")

    print(f"训练完成！最佳验证准确率: {best_acc:.2f}%")

    # 导出 ONNX
    model.load_state_dict(torch.load(model_save_path, map_location='cpu'))
    model.eval()
    model.export_to_onnx(onnx_save_path)
    print(f"ONNX 模型已导出至: {onnx_save_path}")

if __name__ == "__main__":
    train()