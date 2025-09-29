"""
augment_and_split_yolo.py
- Offline augmentation (Albumentations) cho ảnh + nhãn YOLO
- Split dataset thành train/val/test và tạo data.yaml cho Ultralytics
- Sử dụng: chỉnh đường dẫn và chạy các hàm trong phần Example usage

Ghi chú:
- Albumentations sử dụng bbox_params format='yolo' (normalized xywh 0..1).
- Chúng ta giữ nguyên định dạng YOLO cho file nhãn đầu ra.
"""
import os
from pathlib import Path
import random
import shutil
from tqdm import tqdm
import cv2
import albumentations as A
import yaml

# -------------------------
# 1) Hàm đọc/ghi nhãn YOLO
# -------------------------
def read_yolo_label(label_path):
    """
    Trả về list các entry dạng [(cls:int, x:float, y:float, w:float, h:float), ...]
    Nếu file không tồn tại hoặc rỗng -> trả về []
    """
    p = Path(label_path)
    if not p.exists():
        return []
    lines = p.read_text().strip().splitlines()
    out = []
    for l in lines:
        if not l.strip():
            continue
        parts = l.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
        out.append((cls, x, y, w, h))
    return out

def write_yolo_label(label_path, boxes):
    """
    boxes: list of (cls, x, y, w, h) with x,y,w,h normalized 0..1
    """
    p = Path(label_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for (cls, x, y, w, h) in boxes:
        lines.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    p.write_text("\n".join(lines))

# ---------------------------------------
# 2) Một số pipeline augmentation mẫu
# ---------------------------------------
def get_train_augmentations(img_size=640):
    """
    Trả về Albumentations Compose cho training augmentation (on-the-fly / offline)
    Sử dụng bbox_params format='yolo' -> input/output bbox normalized xywh
    """
    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.6, 1.0), ratio=(0.75,1.33), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.4),
        A.Blur(blur_limit=3, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.CoarseDropout(max_holes=1, max_height=int(img_size*0.05), max_width=int(img_size*0.05), p=0.2),
        # nếu dùng nhiều augmentation hơn, bổ sung ở đây
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def get_val_augmentations(img_size=640):
    """Đơn giản resize / center crop cho validation"""
    return A.Compose([
        A.Resize(img_size, img_size)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ---------------------------------------
# 3) Augment 1 ảnh + nhãn và lưu kết quả
# ---------------------------------------
def augment_image_and_label(image_path, label_path, out_image_path, out_label_path, aug, keep_empty=False):
    """
    - image_path: path image (any size)
    - label_path: path .txt YOLO (may be missing)
    - out_image_path: nơi lưu image augment (jpg)
    - out_label_path: nơi lưu label augment (.txt)
    - aug: Albumentations Compose với bbox_params format='yolo'
    - keep_empty: nếu True, khi augmentation xóa hết bbox, vẫn lưu ảnh + file txt rỗng;
                  nếu False và bbox bị xóa hết -> không lưu (bỏ qua)
    Trả về True nếu lưu thành công, False nếu bị bỏ qua.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    labels = read_yolo_label(label_path)
    # Albumentations with format 'yolo' expects normalized (x_center,y_center,w,h)
    bboxes = []
    class_labels = []
    for (cls, x, y, bw, bh) in labels:
        bboxes.append([x, y, bw, bh])
        class_labels.append(int(cls))
    try:
        transformed = aug(image=img, bboxes=bboxes, class_labels=class_labels)
    except Exception as e:
        # nếu augmentation lỗi (ví dụ bbox out-of-range), bỏ qua
        # print("Augment error:", e)
        return False
    out_img = transformed['image']
    out_boxes = transformed.get('bboxes', [])
    out_cls = transformed.get('class_labels', [])

    # nếu không còn bbox và keep_empty False -> bỏ qua
    if (not out_boxes or len(out_boxes) == 0) and not keep_empty:
        return False

    # lưu ảnh và nhãn
    out_image_path = Path(out_image_path)
    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image_path), out_img)

    # out_boxes are in normalized yolo format (because bbox_params.format='yolo')
    boxes_to_write = []
    for cls, box in zip(out_cls, out_boxes):
        x, y, bw, bh = box
        # clip to [0,1] to be safe
        x = max(0.0, min(1.0, float(x)))
        y = max(0.0, min(1.0, float(y)))
        bw = max(0.0, min(1.0, float(bw)))
        bh = max(0.0, min(1.0, float(bh)))
        boxes_to_write.append((int(cls), x, y, bw, bh))
    write_yolo_label(out_label_path, boxes_to_write)
    return True

# ---------------------------------------
# 4) Augment toàn bộ dataset (offline)
# ---------------------------------------
def augment_dataset_offline(images_dir, labels_dir, out_images_dir, out_labels_dir,
                            n_augment_per_image=3, img_size=640, keep_empty=False, only_with_boxes=True,
                            seed=42, aug_fn=None):
    """
    Tạo augmented images + labels offline.
    - images_dir: thư mục chứa ảnh gốc
    - labels_dir: thư mục chứa nhãn gốc (cùng tên .txt)
    - out_images_dir / out_labels_dir: nơi lưu ảnh và nhãn augment
    - n_augment_per_image: số ảnh augment tạo ra cho mỗi ảnh gốc
    - only_with_boxes: nếu True chỉ augment ảnh có ít nhất 1 bbox (tránh tạo nhiều ảnh background)
    - aug_fn: function trả về Albumentations Compose; nếu None dùng get_train_augmentations(img_size)
    Trả về số ảnh augment được tạo.
    """
    random.seed(seed)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_images_dir = Path(out_images_dir)
    out_labels_dir = Path(out_labels_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    if aug_fn is None:
        aug = get_train_augmentations(img_size=img_size)
    else:
        aug = aug_fn(img_size)

    img_paths = sorted([p for p in images_dir.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    count = 0
    for img_path in tqdm(img_paths, desc="Augment images"):
        label_path = labels_dir/img_path.with_suffix('.txt').name
        labels = read_yolo_label(label_path)
        if only_with_boxes and len(labels) == 0:
            # copy original image + label if you want? Here we skip augmenting background images.
            continue
        # create N augmented copies (avoid duplicate filename collisions)
        for i in range(n_augment_per_image):
            out_img_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            out_lbl_name = f"{img_path.stem}_aug{i}.txt"
            ok = augment_image_and_label(img_path, label_path,
                                         out_images_dir/out_img_name,
                                         out_labels_dir/out_lbl_name,
                                         aug, keep_empty=keep_empty)
            if ok:
                count += 1
    return count

# ---------------------------------------
# 5) Chia dataset YOLO theo tỉ lệ (copy files)
# ---------------------------------------
def split_yolo_dataset(images_dir, labels_dir, out_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                       seed=42, copy_files=True):
    """
    Chia dataset image+label thành out_dir/{train,val,test}/{images,labels} theo tỉ lệ.
    - images_dir, labels_dir: thư mục chứa ảnh gốc và nhãn
    - out_dir: thư mục gốc để lưu split
    - copy_files: nếu True copy file, nếu False move (thận trọng)
    Trả về dict with counts.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Tỉ lệ phải cộng = 1"
    random.seed(seed)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)
    # collect images
    imgs = sorted([p for p in images_dir.glob('*') if p.suffix.lower() in ['.jpg','jpeg','.png']])
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_imgs = imgs[:n_train]
    val_imgs = imgs[n_train:n_train+n_val]
    test_imgs = imgs[n_train+n_val:]
    splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
    counts = {}
    for split_name, files in splits.items():
        imdir = out_dir/split_name/'images'
        lbdir = out_dir/split_name/'labels'
        imdir.mkdir(parents=True, exist_ok=True)
        lbdir.mkdir(parents=True, exist_ok=True)
        c = 0
        for im in files:
            src_img = im
            src_lbl = labels_dir/im.with_suffix('.txt').name
            dst_img = imdir/im.name
            dst_lbl = lbdir/im.with_suffix('.txt').name
            if copy_files:
                shutil.copy2(src_img, dst_img)
                if Path(src_lbl).exists():
                    shutil.copy2(src_lbl, dst_lbl)
                else:
                    # tạo file nhãn rỗng nếu không có
                    Path(dst_lbl).write_text('')
            else:
                shutil.move(src_img, dst_img)
                if Path(src_lbl).exists():
                    shutil.move(src_lbl, dst_lbl)
                else:
                    Path(dst_lbl).write_text('')
            c += 1
        counts[split_name] = c
    return counts

# ---------------------------------------
# 6) Sinh data.yaml cho Ultralytics (YOLOv8) từ thư mục out_dir
# ---------------------------------------
def generate_yolo_data_yaml(out_dir, names_list=None, nc=None):
    """
    Tạo file data.yaml tại out_dir/data.yaml
    - out_dir: thư mục gốc chứa train/images, val/images, test/images
    - names_list: optional list tên lớp (['person','car',...]) ; nếu None, infer từ nhãn numeric max
    - nc: optional số lớp; nếu None, infer từ nhãn files
    """
    out_dir = Path(out_dir)
    # infer nc and names if needed
    if nc is None:
        # tìm tất cả labels trong train/labels
        labfiles = list((out_dir/'train'/'labels').glob('*.txt'))
        classes = set()
        for lf in labfiles:
            for line in lf.read_text().splitlines():
                if not line.strip(): continue
                parts = line.strip().split()
                cls = int(parts[0])
                classes.add(cls)
        if len(classes) == 0:
            inferred_nc = 1
        else:
            inferred_nc = max(classes) + 1
    else:
        inferred_nc = nc
    if names_list is None:
        names_list = [str(i) for i in range(inferred_nc)]
    data = {
        'path': str(out_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': inferred_nc,
        'names': names_list
    }
    with open(out_dir/'data.yaml', 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    return out_dir/'data.yaml'

# ---------------------------------------
# 7) Example usage
# ---------------------------------------
if __name__ == "__main__":
    # --- paths (thay đổi theo máy bạn) ---
    # dataset gốc: ảnh + nhãn
    src_images = "dataset/images"   # chứa *.jpg
    src_labels = "dataset/labels"   # chứa *.txt cùng tên

    # nơi lưu augmented (nếu muốn tách riêng)
    aug_images = "dataset_aug/images"
    aug_labels = "dataset_aug/labels"

    # nơi lưu split cuối cùng
    out_split_dir = "dataset_yolo_split"

    # --- 1) Tăng cường offline (tạo thêm ảnh đã augment) ---
    # Lời khuyên: thường chỉ augment THUẦN TRAIN (không augment val/test).
    print("Start augmenting training images (this may take time)...")
    created = augment_dataset_offline(src_images, src_labels, aug_images, aug_labels,
                                      n_augment_per_image=3, img_size=640,
                                      keep_empty=False, only_with_boxes=True, seed=42)
    print(f"Created {created} augmented images")

    # --- 2) Kết hợp gốc + augment (nếu muốn) ---
    # Ở đây ta sẽ combine original images + augmented images trong 1 folder để split
    combined_images = "dataset_combined/images"
    combined_labels = "dataset_combined/labels"
    Path(combined_images).mkdir(parents=True, exist_ok=True)
    Path(combined_labels).mkdir(parents=True, exist_ok=True)
    # copy originals
    for p in Path(src_images).glob('*'):
        if p.suffix.lower() in ['.jpg','jpeg','.png']:
            shutil.copy2(p, Path(combined_images)/p.name)
            l = Path(src_labels)/p.with_suffix('.txt').name
            if l.exists():
                shutil.copy2(l, Path(combined_labels)/l.name)
            else:
                Path(combined_labels)/p.with_suffix('.txt').name.write_text('')
    # copy augmented
    for p in Path(aug_images).glob('*'):
        if p.suffix.lower() in ['.jpg','jpeg','.png']:
            shutil.copy2(p, Path(combined_images)/p.name)
            la = Path(aug_labels)/p.with_suffix('.txt').name
            if la.exists():
                shutil.copy2(la, Path(combined_labels)/la.name)

    # --- 3) Split dataset combined -> train/val/test ---
    counts = split_yolo_dataset(combined_images, combined_labels, out_split_dir,
                                train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                                seed=42, copy_files=True)
    print("Split counts:", counts)

    # --- 4) Generate data.yaml cho Ultralytics ---
    yaml_path = generate_yolo_data_yaml(out_split_dir)
    print("Created data.yaml:", yaml_path)
