# Notebook: Pipeline CNN đầy đủ (Tiếng Việt, chú thích kỹ)
# Mục tiêu: pipeline hoàn chỉnh cho bài toán phân loại ảnh bằng MẠNG CNN
# Bố cục:
# 0) Ghi chú chung, cài đặt
# 1) Chuẩn bị & chia dữ liệu (train/val/test)
# 2) Augmentation & transforms (torchvision)
# 3) DataLoader (ImageFolder)
# 4) Kiến trúc mẫu: SimpleCNN và Fine-tune ResNet
# 5) Vòng huấn luyện + validation + lưu checkpoint
# 6) Đánh giá (accuracy, confusion matrix, precision/recall, F1)
# 7) Export ONNX và hướng dẫn TensorRT
# 8) Ví dụ inference (PyTorch và ONNXRuntime)

# NOTE: Tất cả các cell được phân tách bởi '# %%' để bạn dễ paste vào file .py chạy trong Jupyter/VSCode

# %%
# 0) GHI CHÚ CHUNG
# - Chạy trên máy có GPU: đảm bảo PyTorch được cài đúng phiên bản (CUDA).
# - Cài gói cần thiết (chạy một lần):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install scikit-learn matplotlib onnx onnxruntime tqdm pyyaml
# Nếu muốn export TensorRT cần cài TensorRT/Python binding từ NVIDIA (không phải pip chuẩn trên nhiều hệ).

# %%
# 1) CHIA DỮ LIỆU
# Giả sử dataset gốc có cấu trúc theo class folders:
# dataset_raw/
#   class0/ img1.jpg...
#   class1/ img2.jpg...
# Hàm dưới chia theo tỉ lệ và giữ cấu trúc thư mục.
from pathlib import Path
import random
import shutil

def split_classification_dataset(src_dir, dst_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Chia dataset theo tỉ lệ và copy file vào dst_dir với các thư mục train/val/test/classX"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    classes = [p.name for p in src.iterdir() if p.is_dir()]
    for cls in classes:
        files = [p for p in (src/cls).glob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png']]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]
        for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            out_dir = dst/split_name/cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out_dir/f.name)
    print('Chia dữ liệu xong: mỗi folder có train/val/test')

# Ví dụ:
# split_classification_dataset('dataset_raw','dataset_split',0.8,0.1,0.1)

# %%
# 2) AUGMENTATION & TRANSFORMS (dùng torchvision)
import torch
from torchvision import transforms

def get_classification_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return train_tf, val_tf

# %%
# 3) DATALOADER (dùng ImageFolder cho classification)
from torchvision import datasets
from torch.utils.data import DataLoader

def create_dataloaders(data_dir, img_size=224, batch_size=32, num_workers=4):
    train_tf, val_tf = get_classification_transforms(img_size)
    train_ds = datasets.ImageFolder(Path(data_dir)/'train', transform=train_tf)
    val_ds = datasets.ImageFolder(Path(data_dir)/'val', transform=val_tf)
    test_ds = datasets.ImageFolder(Path(data_dir)/'test', transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    classes = train_ds.classes
    return train_loader, val_loader, test_loader, classes

# %%
# 4) KIẾN TRÚC MẪU
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Mạng CNN đơn giản: phù hợp cho thử nghiệm và dataset nhỏ"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2,2)
        # Nếu img_size=224 -> sau 3 pool: 224->112->56->28
        self.fc1 = nn.Linear(128*28*28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Fine-tune ResNet (khuyến nghị cho bài toán thực tế)
from torchvision import models

def create_resnet_finetune(num_classes, pretrained=True, freeze_backbone=False):
    model = models.resnet50(pretrained=pretrained)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# %%
# 5) VÒNG HUẤN LUYỆN + VALIDATION + LƯU CHECKPOINT
import time
from sklearn.metrics import accuracy_score

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3, checkpoint_path='checkpoint.pth'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    best_acc = 0.0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    for epoch in range(1, epochs+1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        # Lưu checkpoint khi acc tốt hơn
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch}, checkpoint_path)
        print(f'Epoch {epoch}/{epochs} - time: {time.time()-t0:.1f}s - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} (best {best_acc:.4f})')
    return history

# %%
# 6) ĐÁNH GIÁ CHI TIẾT: confusion matrix, precision/recall, f1
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

def evaluate_and_report(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print('Classification report:', report)
    # vẽ confusion matrix đơn giản
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    return cm, report

# %%
# 7) EXPORT sang ONNX & HƯỚNG DẪN TENSORRT
# Export PyTorch model sang ONNX (ví dụ với batch size 1)

def export_to_onnx(model, onnx_path, input_shape=(1,3,224,224), device='cpu', opset=12, dynamic=False):
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    if device!='cpu':
        model.to(device)
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=opset, do_constant_folding=True,
                      input_names=['input'], output_names=['output'], dynamic_axes={'input':{0:'batch_size'}, 'output':{0:'batch_size'}} if dynamic else None)
    print(f'Exported ONNX to {onnx_path}')

# TensorRT: cách đơn giản nhất là dùng trtexec để convert onnx->engine:
# trtexec --onnx=model.onnx --saveEngine=model.trt --fp16 --workspace=4096 --explicitBatch
# Nếu muốn build bằng Python, cần tensorrt Python bindings và code để parse ONNX, set config, build engine (tương tự ví dụ trong YOLO notebook).

# %%
# 8) INFERENCE MẪU (PyTorch và ONNXRuntime)
import numpy as np
import onnxruntime as ort

# PyTorch inference (single image tensor đã normalized, shape (1,3,H,W))
def pytorch_inference(model, image_tensor, device='cpu'):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
    return preds, probs.cpu().numpy()

# ONNXRuntime inference
def onnx_inference(onnx_path, image_numpy):
    # image_numpy: np.array shape (1,3,H,W) float32 (đã normalized)
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: image_numpy})
    probs = outputs[0]
    preds = np.argmax(probs, axis=1)
    return preds, probs

# %%
# KẾT LUẬN
# Đây là notebook đầy đủ cho pipeline CNN (phân loại ảnh) — từ chia dữ liệu, augment, DataLoader, 2 mô hình mẫu,
# training loop, đánh giá, export ONNX và hướng dẫn sơ bộ về TensorRT.
# Nếu bạn muốn mình cập nhật thêm: 
# - Demo chạy thực tế với dataset của bạn (bạn upload đường dẫn/ cấu trúc)
# - Thêm TensorBoard logging / plot loss curves
# - Thêm ví dụ INT8 calibration cho TensorRT
# Hãy trả lời: bạn muốn chạy thử mô hình nào (SimpleCNN hay ResNet50), kích thước ảnh, và có GPU không?
