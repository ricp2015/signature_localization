import pandas as pd
import torch
from torchvision.transforms import functional as F
from PIL import Image
import os
import ast
from tqdm import tqdm

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        dataset = {}
        image_dir = "data/raw/signverod_dataset/images/"
        image_info_path = "data/raw/fixed_dataset/updated_image_ids.csv"
        annotations_path = "data/raw/fixed_dataset/full_data.csv"
        image_info = pd.read_csv(image_info_path)
        annotations = pd.read_csv(annotations_path)
        signature_annotations = annotations[annotations['category_id'] == 1]
        for _, row in tqdm(signature_annotations.iterrows(), total=len(signature_annotations)):
            bbox = ast.literal_eval(row['bbox'])
            image_id = row['image_id']
            # Find the corresponding image info
            image_info_row = image_info[image_info['id'] == image_id].iloc[0]
            file_name = image_dir+image_info_row['file_name']
            img_height = image_info_row['height']
            img_width = image_info_row['width']
            if image_id not in dataset:
                dataset[image_id] = {"name":file_name, "boxes":[]}
            # Calculate bounding box in pixel coordinates
            x_min = int(bbox[0] * img_width)
            y_min = int(bbox[1] * img_height)
            x_max = int((bbox[0] + bbox[2]) * img_width)
            y_max = int((bbox[1] + bbox[3]) * img_height)
            dataset[image_id]["boxes"].append((x_min, y_min, x_max, y_max))
        self.dataset = [y for x, y in sorted(dataset.items())]
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Carica immagine e annotazioni
        img_path = self.dataset[idx]['name']
        boxes = self.dataset[idx]['boxes']

        image = Image.open(img_path).convert("RGB")
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones(len(boxes), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            image = self.transforms(image)

        return F.to_tensor(image), target


from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def train():
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = torch.utils.data.DataLoader(
            CustomDataset(),
            batch_size=64,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.train()
    for epoch in range(5):
        for images, targets in train_loader:
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {losses.item()}")
