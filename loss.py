import torch
from utils import intersecton_over_union
import numpy as np

def loss_fn(predictions, target_whole, BATCH_SIZE):
    # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
    # prediction은 예측하는 것, target은 ground truth
    lambda_coord = 5
    lambda_noobj = 0.5
    predictions = predictions.reshape(-1,7,7,30)  
    
    threshold = 0.5  
    loss = 0
    # print("BATCH_SIZE", BATCH_SIZE)
    for batch_index in range(predictions.shape[0]):
        
        # print(batch_index)
        # for batch_index in range(1):
        target = target_whole[batch_index, ...]
        # print("target.shape", target.shape)
        prediction_batch = predictions[batch_index,...]
        confidence = target[...,20]
        # print(confidence)
        where = np.nonzero(confidence > threshold)
        exist_box = confidence.unsqueeze(2) # unsqueeze 하기전에는 [7, 7]
        # print(exist_box.shape)
        # print("(1-exist_box).shape : ", (1-exist_box).shape)
        no_object_loss = torch.sum(((1-exist_box) * (target[...,20]-prediction_batch[...,20]))**2)
        no_object_loss += torch.sum(((1-exist_box) * (target[...,20]-prediction_batch[...,25]))**2)
        loss = loss + lambda_noobj * no_object_loss 
        for i in range(len(where)):
            x_grid = where[i,0]
            y_grid = where[i,1]
            # class_index = torch.argmax(target[x_grid,y_grid,:20], dim=0)
            
            iou_b1 = intersecton_over_union(prediction_batch[x_grid,y_grid,21:25], target[x_grid,y_grid,21:25])
            # print("iou_b1 : ", iou_b1)
            iou_b2 = intersecton_over_union(prediction_batch[x_grid,y_grid,26:30], target[x_grid,y_grid,21:25])
            # print("iou_b2 : ", iou_b2)
            if iou_b1 > iou_b2:
                # b1이 iou가 더 높아서 responsible
                loss_coord = ((target[x_grid, y_grid, 21] - prediction_batch[x_grid, y_grid, 21])**2 
                + (target[x_grid, y_grid, 22] - prediction_batch[x_grid, y_grid, 22])**2)
                loss_coord = loss_coord + ((torch.sqrt(target[x_grid, y_grid, 23]) - torch.sign(prediction_batch[x_grid, y_grid, 24]) * torch.sqrt(abs(prediction_batch[x_grid, y_grid, 23])))**2 
                + (torch.sqrt(target[x_grid, y_grid, 24])- torch.sign(prediction_batch[x_grid, y_grid, 24]) * torch.sqrt(abs(prediction_batch[x_grid, y_grid, 24])))**2) 

                object_loss = (target[x_grid, y_grid, 20] - prediction_batch[x_grid, y_grid, 20])**2
                class_loss = torch.sum((target[x_grid, y_grid, :20] - prediction_batch[x_grid, y_grid, :20])**2)
                
                loss = loss + lambda_coord * loss_coord + object_loss + class_loss
                # print("1 is bigger")
                # print(loss)
            else:
                loss_coord = ((target[x_grid, y_grid, 21] - prediction_batch[x_grid, y_grid, 26])**2 
                + (target[x_grid, y_grid, 22] - prediction_batch[x_grid, y_grid, 27])**2)
                loss_coord = loss_coord + ((torch.sqrt(target[x_grid, y_grid, 23]) - torch.sqrt(abs(prediction_batch[x_grid, y_grid, 28])))**2 
                + (torch.sqrt(target[x_grid, y_grid, 24])- torch.sqrt(abs(prediction_batch[x_grid, y_grid, 29])))**2) 

                object_loss = (target[x_grid, y_grid, 20] - prediction_batch[x_grid, y_grid, 25])**2
                class_loss = torch.sum((target[x_grid, y_grid, :20] - prediction_batch[x_grid, y_grid, :20])**2)
                
                loss = loss + lambda_coord * loss_coord + object_loss + class_loss
                # print("2 is bigger")
                # print(loss)

    return loss             
