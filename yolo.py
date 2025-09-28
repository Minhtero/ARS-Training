# Notebook: Pipeline đầy đủ YOLOv8 (Tiếng Việt, chú thích nhiều)
# Các cell được phân tách bằng '# %%' để tiện chạy trong Jupyter/VSCode
# MỤC LỤC
# 0) Ghi chú chung và cài đặt
# 1) Chia dữ liệu (train/val/test) cho định dạng YOLO
# 2) (Tuỳ chọn) Chuyển Pascal VOC -> YOLO
# 3) Augmentation (on-the-fly và offline) bằng Albumentations
# 4) PyTorch Dataset / DataLoader cho nhãn kiểu YOLO
# 5) Huấn luyện sử dụng Ultralytics YOLOv8 (giao diện đơn giản và mạnh)
# 6) Đánh giá (validation, mAP) và xem kết quả
# 7) Export sang ONNX và chuyển ONNX -> TensorRT
# 8) Inference: ONNXRuntime và TensorRT (mẫu cơ bản)
# 9) Ghi chú triển khai, lỗi thường gặp và mẹo

# %%
# 0) GHI CHÚ CHUNG
# - File này giả định dataset ảnh có nhãn theo định dạng YOLO (.txt cùng tên ảnh)
# - Mỗi dòng trong file nhãn: <class_id> <x_center> <y_center> <width> <height> (tất cả chuẩn hoá 0..1)
# - Bạn có thể chạy từng cell trong Jupyter. Một số cell cần quyền admin hoặc đã được cài sẵn (ví dụ TensorRT).
# - Một số bước (cài đặt TensorRT, trtexec) cần thao tác ngoài Python.

# Cài đặt gói (chạy một lần nếu cần):
# pip install ultralytics==8.* albumentations opencv-python-headless pycocotools onnx onnxruntime tqdm pyyaml
# Nếu muốn dùng TensorRT Python API, cài bản tương thích do NVIDIA cung cấp (không phải pip bình thường trên nhiều hệ).

# %%
import os
from pathlib import Path
import random
import shutil
from tqdm import tqdm

# %%
# 1) Chia dataset YOLO thành train/val/test
# Hàm này copy ảnh và file nhãn tương ứng vào thư mục đích theo cấu trúc:
# dst_dir/
#   train/images/*.jpg
#   train/labels/*.txt
#   val/images/*
#   val/labels/*
#   test/images/*
#   test/labels/*
# Đồng thời sinh file data.yaml phù hợp cho Ultralytics (chứa path, nc, names)

def split_yolo_dataset(src_dir, dst_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Chia dữ liệu ảnh + nhãn dạng YOLO theo tỉ lệ.
    Nếu ảnh không có file nhãn tương ứng, sẽ tạo file txt rỗng (không có object).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Tỉ lệ phải cộng đúng 1.0"
    random.seed(seed)
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    imgs = [p for p in src_dir.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    imgs = sorted(imgs)
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = imgs[:n_train]
    val = imgs[n_train:n_train+n_val]
    test = imgs[n_train+n_val:]

    def copy_split(files, split_name):
        imdir = dst_dir/split_name/'images'
        lbdir = dst_dir/split_name/'labels'
        imdir.mkdir(parents=True, exist_ok=True)
        lbdir.mkdir(parents=True, exist_ok=True)
        for im in files:
            shutil.copy2(im, imdir/im.name)
            lb = im.with_suffix('.txt')
            if lb.exists():
                shutil.copy2(lb, lbdir/lb.name)
            else:
                # tạo file txt rỗng nếu không có nhãn (nên tránh nhưng tiện khi test)
                (lbdir/im.with_suffix('.txt').name).write_text('')

    copy_split(train, 'train')
    copy_split(val, 'val')
    copy_split(test, 'test')

    # infer số lớp từ folder train/labels
    nc = _infer_num_classes(dst_dir/'train'/'labels')
    # tạo data.yaml cho ultralytics
    data_yaml = {
        'path': str(dst_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': nc,
        'names': [str(i) for i in range(nc)]
    }
    import yaml
    with open(dst_dir/'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    print(f"Split xong. train={len(train)}, val={len(val)}, test={len(test)}. data.yaml viết vào {dst_dir/'data.yaml'}")

# Helper: đếm số lớp từ folder label
def _infer_num_classes(labels_dir):
    labels_dir = Path(labels_dir)
    classes = set()
    if not labels_dir.exists():
        return 1
    for txt in labels_dir.glob('*.txt'):
        for line in txt.read_text().splitlines():
            s = line.strip().split()
            if len(s) >= 1 and s[0].lstrip('-').isdigit():
                classes.add(int(s[0]))
    return max(classes)+1 if classes else 1

# Ví dụ sử dụng:
# split_yolo_dataset('dataset_origin_images', 'dataset_yolo_split', 0.8, 0.1, 0.1)

# %%
# 2) (Tuỳ chọn) Chuyển Pascal VOC (XML) sang YOLO
# Nếu dataset gốc ở dạng VOC (XML), dùng hàm này: cần cung cấp list 'classes' (tên các lớp)
import xml.etree.ElementTree as ET

def voc_to_yolo(voc_dir, out_img_dir, out_label_dir, classes):
    """Chuyển file XML của VOC sang file .txt định dạng YOLO.
    voc_dir: thư mục chứa .xml và thư mục JPEGImages/ chứa ảnh
    classes: list tên lớp (ví dụ ['apple','banana']). Nếu object có tên không nằm trong classes thì bỏ qua.
    """
    voc_dir = Path(voc_dir)
    out_img_dir = Path(out_img_dir); out_label_dir = Path(out_label_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    for xml in voc_dir.glob('*.xml'):
        tree = ET.parse(xml)
        root = tree.getroot()
        file_name = root.find('filename').text
        size = root.find('size')
        w = float(size.find('width').text)
        h = float(size.find('height').text)
        img_path = xml.parent/'JPEGImages'/file_name
        if img_path.exists():
            shutil.copy2(img_path, out_img_dir/file_name)
        txt_lines = []
        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            b = obj.find('bndbox')
            xmin = float(b.find('xmin').text)
            ymin = float(b.find('ymin').text)
            xmax = float(b.find('xmax').text)
            ymax = float(b.find('ymax').text)
            x_center = ((xmin + xmax) / 2.0) / w
            y_center = ((ymin + ymax) / 2.0) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            txt_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
        (out_label_dir/xml.with_suffix('.txt').name).write_text(''.join(txt_lines))

# %%
# 3) Augmentation - dùng Albumentations
# - Có thể augment offline (tạo file ảnh mới) hoặc on-the-fly trong Dataset
# - Ở đây có 2 hàm: get_train_transforms và get_valid_transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

def get_train_transforms(img_size=640):
    # Áp dụng một số augmentation phổ biến cho object detection
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.6, 1.0), ratio=(0.9,1.1), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Blur(p=0.2),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(max_holes=1, max_height=int(img_size*0.05), max_width=int(img_size*0.05), p=0.2)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def get_valid_transforms(img_size=640):
    return A.Compose([A.Resize(img_size, img_size)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Offline augmentation (mở rộng dataset, lưu ảnh + label mới)
def augment_and_save(image_path, label_path, out_img_dir, out_lbl_dir, n_augment=5, img_size=640):
    img = cv2.imread(str(image_path))
    h0, w0 = img.shape[:2]
    # đọc nhãn yolo
    bboxes = []
    class_labels = []
    if Path(label_path).exists():
        for l in open(label_path).read().splitlines():
            if not l.strip():
                continue
            c, x, y, bw, bh = l.split()
            bboxes.append([float(x), float(y), float(bw), float(bh)])
            class_labels.append(int(c))
    aug = get_train_transforms(img_size=img_size)
    out_img_dir = Path(out_img_dir); out_lbl_dir = Path(out_lbl_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_augment):
        try:
            transformed = aug(image=img, bboxes=bboxes, class_labels=class_labels)
        except Exception as e:
            # nếu augmentation thất bại (ví dụ bbox bị loại hết), bỏ qua
            continue
        out_img = transformed['image']
        out_boxes = transformed['bboxes']
        out_classes = transformed['class_labels']
        out_name = f"{Path(image_path).stem}_aug{i}{Path(image_path).suffix}"
        cv2.imwrite(str(out_img_dir/out_name), out_img)
        txt_lines = []
        for cls, box in zip(out_classes, out_boxes):
            x, y, bw, bh = box
            txt_lines.append(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
        (out_lbl_dir/Path(out_name).with_suffix('.txt').name).write_text(''.join(txt_lines))

# %%
# 4) PyTorch Dataset cho nhãn YOLO (on-the-fly augmentation)
import torch
from torch.utils.data import Dataset, DataLoader

class YoloDataset(Dataset):
    """Dataset trả về (image_tensor, boxes_tensor)
    - image_tensor: shape (C,H,W) float32, giá trị đã chia 255
    - boxes_tensor: Nx5 (class, x_center, y_center, w, h) (tất cả normalized 0..1). Nếu không có obj -> tensor shape (0,5)
    """
    def __init__(self, images_dir, labels_dir, transforms=None, img_size=640):
        self.images = sorted([p for p in Path(images_dir).glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))[:,:,::-1]  # BGR->RGB
        h0, w0 = img.shape[:2]
        label_path = self.labels_dir / img_path.with_suffix('.txt').name
        bboxes = []
        class_labels = []
        if label_path.exists():
            for l in label_path.read_text().splitlines():
                if not l.strip():
                    continue
                parts = l.split()
                cls = int(parts[0])
                x, y, bw, bh = map(float, parts[1:5])
                bboxes.append([x, y, bw, bh])
                class_labels.append(cls)
        # nếu có transforms: Albumentations yêu cầu bboxes theo format đã chọn (ở đây 'yolo')
        if self.transforms:
            transformed = self.transforms(image=img, bboxes=bboxes, class_labels=class_labels)
            img = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
        # chuẩn hoá và đổi chuẩn cho PyTorch
        img = img.transpose(2,0,1).astype('float32')/255.0
        if len(bboxes)==0:
            boxes = torch.zeros((0,5), dtype=torch.float32)
        else:
            boxes = torch.tensor([[cls, *b] for cls,b in zip(class_labels, bboxes)], dtype=torch.float32)
        return torch.tensor(img), boxes

# Collate function đơn giản để batch các mẫu có số boxes khác nhau
# Ultralytics tự xử lý batching khi dùng model.train - nên nếu dùng DataLoader tùy chỉnh cần collate

def yolo_collate(batch):
    imgs = [item[0] for item in batch]
    boxes = [item[1] for item in batch]
    imgs = torch.stack(imgs, dim=0)
    return imgs, boxes

# Ví dụ DataLoader:
# train_ds = YoloDataset('dataset_yolo_split/train/images','dataset_yolo_split/train/labels', transforms=get_train_transforms(640))
# train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=yolo_collate)

# %%
# 5) Huấn luyện với Ultralytics YOLOv8
# Ultralytics cung cấp API cao cấp: YOLO('yolov8n.pt') -> model.train(...)
# Lợi ích: xử lý augment nội bộ (mosaic, mixup), mixed precision, logging, callbacks

from ultralytics import YOLO

# Tạo/ tải model
# 'yolov8n.pt' là model nhỏ (nano). Thay bằng 'yolov8s.pt','yolov8m.pt' tuỳ GPU/nhu cầu
model = YOLO('yolov8n.pt')

# Thí dụ train (nhớ sửa đường dẫn data.yaml nếu cần):
# model.train(data='dataset_yolo_split/data.yaml', epochs=50, imgsz=640, batch=16, device=0, workers=4, augment=True)
# Một số tham số quan trọng:
# - data: đường dẫn tới data.yaml
# - epochs: số epoch
# - imgsz: kích thước ảnh huấn luyện
# - batch: kích thước batch tổng (tùy GPU)
# - device: 0,1,... hoặc 'cpu'
# - workers: số worker DataLoader
# - augment: True để bật augment nội bộ (mosaic, mixup)

# Lưu ý: Nếu GPU nhỏ (ví dụ 4GB), giảm batch size hoặc dùng yolov8n

# %%
# 6) Đánh giá (validation) và xem kết quả
# Sau khi train xong, ultralytics lưu weights tại 'runs/detect/train/weights/best.pt' (tuỳ đường dẫn)
# Cách load model đã train và đánh giá:

# model = YOLO('runs/detect/train/weights/best.pt')
# results = model.val()  # in kết quả mAP@0.5 và mAP@0.5:0.95
# print(results)

# Hoặc dùng model.predict để dự đoán trên thư mục val:
# preds = model.predict(source='dataset_yolo_split/val/images', imgsz=640, conf=0.25)
# model.predict trả về list Result thay vì tensor thô; bạn có thể xuất json hoặc plot

# %%
# 7) Export sang ONNX và chuyển sang TensorRT
# 7.1 Export ONNX (Ultralytics hỗ trợ hàm export đơn giản)
# Sau khi load model tốt nhất (best.pt):
# model = YOLO('runs/detect/train/weights/best.pt')
# model.export(format='onnx')  # sẽ sinh file .onnx tương ứng
# Tham số thêm: opset (ví dụ 12), dynamic (True/False)
# model.export(format='onnx', opset=12, dynamic=True)

# 7.2 Chuyển ONNX -> TensorRT
# Cách đơn giản nhất: dùng trtexec (CLI) (được cài cùng TensorRT)
# Ví dụ (shell):
# trtexec --onnx=best.onnx --saveEngine=best.trt --fp16 --workspace=4096 --explicitBatch
# - --fp16 bật FP16 nếu driver và card hỗ trợ
# - --workspace: MB hoặc MiB tùy trtexec (ở đây để integer: 4096MB)

# 7.3 (Tuỳ chọn) Build engine TensorRT bằng Python API (cần tensorrt Python lib)
# Mình để mẫu xây dựng engine cơ bản (cần tensorrt cài sẵn):

def build_engine_from_onnx(onnx_file_path, engine_file_path, fp16=False, max_workspace_size=1<<30):
    try:
        import tensorrt as trt
    except Exception as e:
        raise RuntimeError('Cần cài tensorrt Python package để chạy hàm này. Xem hướng dẫn NVIDIA.')
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError('Parse ONNX thất bại')
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_engine(network, config)
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    print(f'Đã ghi engine TensorRT: {engine_file_path}')

# %%
# 8) Inference mẫu
# 8.1 ONNXRuntime inference mẫu (CPU hoặc GPU nếu có CUDA provider)
import onnxruntime as ort
import numpy as np

def run_onnx_inference(onnx_path, input_image_numpy):
    # input_image_numpy: numpy array shape (N,C,H,W) float32
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: input_image_numpy})
    return out

# 8.2 TensorRT runtime (mẫu - yêu cầu pycuda và tensorrt)
# Lưu ý: code cho TensorRT cần biết binding indices, shapes; ở đây chỉ cung cấp skeleton

def run_trt_engine(engine_path, input_numpy):
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except Exception as e:
        raise RuntimeError('Cần tensorrt và pycuda để chạy inference TensorRT')
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    # Lấy thông tin binding để cấp phát bộ nhớ
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # cấp phát bộ nhớ device
        # ... (thực hiện theo binding index: pycuda.driver.mem_alloc, cuda.memcpy_htod_async,...)
    print('Mẫu chạy TensorRT: cần triển khai chi tiết theo shape/binding của model')

# %%
# 9) Ghi chú, mẹo và lỗi thường gặp
# - Nếu export ONNX lỗi: thử giảm opset (ví dụ 12), bật dynamic axes hoặc dùng onnx-simplifier
# - TensorRT không hỗ trợ một số op: cần dùng ONNX simplifier hoặc custom plugin
# - INT8 tối ưu nhất về tốc độ nhưng cần calibration dataset để ra kết quả chính xác
# - Khi training gặp overfit: tăng augmentation, giảm learning rate, early stopping hoặc dùng dữ liệu nhiều hơn
# - Theo dõi GPU memory: nếu OOM, giảm batch_size hoặc dùng model nhỏ hơn (yolov8n)

# KẾT THÚC
# Lưu ý: notebook này là một khuôn mẫu chi tiết bằng tiếng Việt. Nếu thầy muốn mình thêm: 
# - ví dụ data.yaml cho nhiều lớp có tên cụ thể
# - cell chạy thực tế model.train với logging/plot (TensorBoard hoặc loss curves)
# - chi tiết code TensorRT inference (cấp phát buffer đầy đủ) cho model của thầy
# Hãy nói rõ lựa chọn (hoặc gửi đường dẫn dataset + số lớp) và mình sẽ bổ sung ngay vào notebook.
