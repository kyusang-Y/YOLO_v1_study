import torch
import torch.nn as nn
from utils import intersecton_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5   

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        # prediction은 예측하는 것, target은 ground truth
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        iou_b1 = intersecton_over_union(predictions[...,21:25], target[...,21:25])
        iou_b2 = intersecton_over_union(predictions[...,26:30], target[...,21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)   # 밑에다가 concat

        iou_maxes, bestbox = torch.max(ious, dim=0)
        # iou_maxes는 iou가 큰 부분의 x y w h
        # bestbox는 b1이 더 컸으면 0, b2가 더 컸으면 1

        exists_box = target[..., 20].unsqueeze(3)   
        # object가 존재하는 부분만 1 나머지 0, unsqueeze는 dim 사라지는 것 보상

        #### box coordinate ####
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30] +
                (1- bestbox) * predictions[...,21:25]
            )
        )

        box_targets = exists_box * target[...,21:25]
        
        box_predictions[...,2:4] = torch.sign(box_predictions[..., 2:4])*torch.sqrt(
            torch.abs(box_predictions[...,2:4] + 1e-6)
            )

        box_targets[..., 2:4] = torch.sqrt(box_targets[...,2:4])

        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2),
        torch.flatten(box_targets, end_dim=-2))


        ### object가 있는데 못잡는 경우 ###
        pred_box = (
            bestbox * predictions[..., 25:26] + (1-bestbox) * predictions[..., 20:21]
        )
        
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box* target[..., 20:21]),
        )

        ### object가 없는데 잡는 경우 ###
        no_object_loss = self.mse(
            torch.flatten((1-exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1-exists_box) * target[...,20:21], start_dim=1)
        )
        
        # batch size는 건들이지 않는
        no_object_loss = no_object_loss + self.mse(
            torch.flatten((1-exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1-exists_box) * target[...,20:21], start_dim=1)
        )

        ### 잘못된 class로 분류하는 경우 ####
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        loss = (
            self.lambda_coord * box_loss +
            object_loss + self.lambda_noobj * no_object_loss + class_loss)
        return loss