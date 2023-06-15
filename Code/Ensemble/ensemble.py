import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.single_stage_detector import SingleStageDetector
from models.multi_stage_detector import MultiStageDetector

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleStageDetector(nn.Module):
    def __init__(self, num_classes):
        super(SingleStageDetector, self).__init__()

        # Define the backbone network
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Define the object detection head
        self.object_detection_head = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.object_detection_head(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x


class MultiStageDetector(nn.Module):
    def __init__(self, num_classes):
        super(MultiStageDetector, self).__init__()

        # Define the backbone network
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Define the region proposal network (RPN)
        self.rpn = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.rpn_box_head = nn.Conv2d(256, 4, kernel_size=1, stride=1, padding=0)
        self.rpn_class_head = nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0)

        # Define the region of interest (RoI) pooling layer
        self.roi_pool = nn.AdaptiveMaxPool2d((7, 7))

        # Define the object detection head
        self.object_detection_head = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes + 4)
        )

    def forward(self, x):
        # Backbone network
        x = self.backbone(x)

        # Region proposal network (RPN)
        rpn_out = self.rpn(x)
        rpn_box_pred = self.rpn_box_head(rpn_out)
        rpn_class_pred = self.rpn_class_head(rpn_out)

        # Region of interest (RoI) pooling layer
        rois = []
        for i in range(rpn_class_pred.shape[0]):
            # Generate candidate regions based on the predicted bounding boxes
            candidates = self.generate_candidates(rpn_box_pred[i], rpn_class_pred[i])
            rois.append(self.roi_pool(x[i:i+1, :, candidates[:, 0]:candidates[:, 2], candidates[:, 1]:candidates[:, 3]]))
        rois = torch.cat(rois)

        # Object detection head
        x = rois.view(rois.shape[0], -1)
        x = self.object_detection_head(x)

        return x

    def generate_candidates(self, box_pred, class_pred):
        # Convert the predicted bounding box coordinates from offsets to absolute values
        box_pred = box_pred.permute(1, 2, 0).contiguous().view(-1, 4)
        anchor_boxes = self.generate_anchor_boxes(box_pred.shape[0], box_pred.device)
        box_pred = self.decode_box_predictions(anchor_boxes, box_pred)

        # Apply non-maximum suppression to remove overlapping bounding boxes
        class_scores = class_pred.permute(1, 2, 0).contiguous().view(-1, 2)[:, 1]
        keep_indices = torchvision.ops.nms(box_pred, class_scores, iou_threshold=0.5)

        return torch.cat([box_pred[keep_indices]])


# Define your data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load your dataset
train_dataset = datasets.CocoDetection(root='DATA/annotations/', annFile='path/to/annotations.json', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Instantiate your single-stage detector and multi-stage detector
single_stage_detector = SingleStageDetector(num_classes=80)
multi_stage_detector = MultiStageDetector(num_classes=80)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(single_stage_detector.parameters()) + list(multi_stage_detector.parameters()), lr=0.001)

# Train your ensemble model
for epoch in range(10):
    for images, targets in train_loader:
        # Forward pass through both detectors
        single_stage_output = single_stage_detector(images)
        multi_stage_output = multi_stage_detector(images)

        # Compute the loss for both detectors
        single_stage_loss = criterion(single_stage_output, targets)
        multi_stage_loss = criterion(multi_stage_output, targets)

        # Compute the total loss
        total_loss = single_stage_loss + multi_stage_loss

        # Backward pass and update parameters
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print training progress
        print(
            f"Epoch: {epoch + 1}/{10}, Single-stage Loss: {single_stage_loss.item()}, Multi-stage Loss: {multi_stage_loss.item()}, Total Loss: {total_loss.item()}")