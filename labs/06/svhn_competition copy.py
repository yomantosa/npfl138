#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss, batched_nms

import bboxes_utils
import npfl138
npfl138.require_version("2425.6.1")
from npfl138.datasets.svhn import SVHN

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class SVHNDetectionDataset(npfl138.TransformedDataset):
    def __init__(self, dataset, image_size=224, transform=None):
        self.base = dataset
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        image = item['image']  # [3, H, W], uint8
        bboxes = item['bboxes'].clone().float()  # [N, 4]
        labels = item['classes']  # [N]

        orig_h, orig_w = image.shape[1:]
        scale_y = self.image_size / orig_h
        scale_x = self.image_size / orig_w

        # Resize image using bilinear interpolation (and normalize dtype)
        image = image.unsqueeze(0).float() / 255.0  # [1, 3, H, W]
        image = torch.nn.functional.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        image = image.squeeze(0)

        # Scale bounding boxes
        bboxes[:, 0] *= scale_y  # top
        bboxes[:, 1] *= scale_x  # left
        bboxes[:, 2] *= scale_y  # bottom
        bboxes[:, 3] *= scale_x  # right

        if self.transform:
            image = self.transform(image)

        return image, (bboxes, labels)
        
class RetinaNetSVHN(npfl138.TrainableModule):
    def __init__(self, backbone, anchors, num_classes=10, score_threshold=0.5, iou_threshold=0.5):
        super().__init__()
        self.backbone = backbone
        self.anchors = anchors
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        self.cls_head = torch.nn.Sequential(
            torch.nn.Conv2d(1280, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, num_classes, 1),
        )
        self.box_head = torch.nn.Sequential(
            torch.nn.Conv2d(1280, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 4, 1),
        )

    def forward(self, x):
        features, _ = self.backbone.forward_intermediates(x)
        f = features[-1]
        cls_logits = self.cls_head(f)
        box_deltas = self.box_head(f)
        return cls_logits, box_deltas

    def compute_loss(self, y_pred, y, *xs):
        cls_logits, box_deltas = y_pred  # Each is [B, A, C] and [B, A, 4] after reshape
        bboxes_list, classes_list = y  # Ground-truth data from dataloader
        device = self.device

        B, A = cls_logits.shape[:2]
        total_cls_loss, total_box_loss = 0.0, 0.0
        losses = []

        for i in range(B):
            valid = classes_list[i] != -1
            if valid.sum() == 0:
                continue

            # Compute target labels and regression deltas
            anchor_classes, anchor_targets = bboxes_utils.bboxes_training(
                self.anchors.cpu(),
                classes_list[i][valid].cpu(),
                bboxes_list[i][valid].float().cpu(),
                iou_threshold=0.5
            )
            
            # Compute classification loss (focal loss)
            target_cls = torch.nn.functional.one_hot(anchor_classes, num_classes=self.num_classes + 1).float().to(device)
            cls_loss = sigmoid_focal_loss(cls_logits[i], target_cls[:, 1:], reduction="mean")

            # Compute box regression loss (Smooth L1) for foreground only
            fg_mask = anchor_classes > 0
            if fg_mask.any():
                box_loss = torch.nn.SmoothL1Loss()(box_deltas[i][fg_mask], anchor_targets[fg_mask].to(device))
            else:
                box_loss = torch.tensor(0.0, device=device)

            total_cls_loss += cls_loss.item()
            total_box_loss += box_loss.item()
            losses.append(cls_loss + box_loss)

        if len(losses) > 0:
            loss = torch.stack(losses).mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        self.log("cls_loss", total_cls_loss / max(len(losses), 1))
        self.log("box_loss", total_box_loss / max(len(losses), 1))
        return loss


    def training_step(self, batch):
        images, (bboxes_list, classes_list) = batch
        images = images.to(self.device)

        cls_logits, box_deltas = self(images)

        B, A = images.shape[0], self.anchors.shape[0]
        cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(B, A, -1)
        box_deltas = box_deltas.permute(0, 2, 3, 1).reshape(B, A, 4)

        total_cls_loss, total_box_loss = 0.0, 0.0
        losses = []

        for i in range(B):
            valid = classes_list[i] != -1
            if valid.sum() == 0:
                continue

            # Get anchor_classes and anchor_targets from bboxes_utils
            anchor_classes, anchor_targets = bboxes_utils.bboxes_training(
                self.anchors, classes_list[i][valid], bboxes_list[i][valid].float(), iou_threshold=0.5
            )

            # Now compute the loss for this image
            cls_loss, box_loss = self.compute_loss(cls_logits[i], box_deltas[i], anchor_classes, anchor_targets)

            total_cls_loss += cls_loss.item()
            total_box_loss += box_loss.item()
            losses.append(cls_loss + box_loss)

        if len(losses) > 0:
            loss = torch.stack(losses).mean()
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log("cls_loss", total_cls_loss / max(len(losses), 1))
        self.log("box_loss", total_box_loss / max(len(losses), 1))
        return loss

    def predict(self, image):
        self.eval()
        with torch.no_grad():
            cls_logits, box_deltas = self(image.unsqueeze(0))
            cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            box_deltas = box_deltas.permute(0, 2, 3, 1).reshape(-1, 4)

            probs = torch.softmax(cls_logits, dim=-1)
            scores, labels = torch.max(probs, dim=-1)
            keep = scores > self.score_threshold
            if keep.sum() == 0:
                return [], torch.zeros((0, 4), device=image.device)

            scores = scores[keep]
            labels = labels[keep]
            boxes = bboxes_utils.bboxes_from_rcnn(self.anchors[keep], box_deltas[keep])
            keep_idx = batched_nms(boxes, scores, labels, iou_threshold=self.iou_threshold)
            return labels[keep_idx], boxes[keep_idx]

            
            
def generate_anchors(grid_size=7, image_size=224, anchor_size=32):
    step = image_size // grid_size
    anchors = []
    for y in range(grid_size):
        for x in range(grid_size):
            cy, cx = y * step + step // 2, x * step + step // 2
            top = cy - anchor_size // 2
            left = cx - anchor_size // 2
            bottom = cy + anchor_size // 2
            right = cx + anchor_size // 2
            anchors.append([top, left, bottom, right])
    return torch.tensor(anchors, dtype=torch.float32)

def detection_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    max_objects = max(item[1][0].shape[0] for item in batch)

    padded_bboxes = []
    padded_classes = []

    for bboxes, classes in [item[1] for item in batch]:
        pad_len = max_objects - bboxes.shape[0]
        padded_bboxes.append(torch.cat([bboxes, torch.zeros(pad_len, 4)]))
        padded_classes.append(torch.cat([classes, torch.full((pad_len,), -1)]))

    bboxes_tensor = torch.stack(padded_bboxes)
    classes_tensor = torch.stack(padded_classes)

    return images, (bboxes_tensor, classes_tensor)

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, SIZE, SIZE]` tensor of `torch.uint8` values in [0-255] range,
    # - "classes", a `[num_digits]` PyTorch vector with classes of image digits,
    # - "bboxes", a `[num_digits, 4]` PyTorch vector with bounding boxes of image digits.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    svhn = SVHN(decode_on_demand=False)

    # Load the EfficientNetV2-B0 model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = efficientnetv2_b0.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1280, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], 
                     std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    # TODO: Create the model and train it.
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    anchors = generate_anchors().to(device)
    model = RetinaNetSVHN(efficientnetv2_b0, anchors=anchors).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    model.configure(
        optimizer=optimizer,
        logdir=args.logdir,
    )

    # Prepare dataset and dataloaders
    train_dataset = SVHNDetectionDataset(svhn.train, transform=preprocessing)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=detection_collate_fn)

    # Train model
    model.fit(train_loader, epochs=args.epochs)


    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    test_dataset = SVHNDetectionDataset(svhn.test, transform=preprocessing)
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
        for example in test_dataset:
            image = example["image"].to(device)
            predicted_classes, predicted_bboxes = model.predict(image)
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [int(label.item())] + list(map(float, bbox.tolist()))
            print(*output, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
