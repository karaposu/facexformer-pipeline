
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import InterpolationMode
import argparse
from math import cos, sin
from PIL import Image
from network import FaceXFormer
from facenet_pytorch import MTCNN
import os
# from argparse import  args
from facexformer_pipeline.utils import denorm_points, unnormalize, adjust_bbox, visualize_head_pose, visualize_landmarks, visualize_mask

def process_visibility(visibility_output):
    probs = torch.sigmoid(visibility_output[0])
    preds = (probs >= 0.5).float()
    pred = preds.tolist()
    pred_str = [str(int(b)) for b in pred]
    return pred_str

def task_gender(age_output,gender_output,race_output ):
    age_preds = torch.argmax(age_output, dim=1)[0]
    gender_preds = torch.argmax(gender_output, dim=1)[0]
    race_preds = torch.argmax(race_output, dim=1)[0]
    predictions_dict = {
        "age": age_preds.item(),
        "gender": gender_preds.item(),
        "race": race_preds.item()
    }

    # Now predictions_dict contains all the predictions
    return predictions_dict
def task_attributes(attribute_output):
    probs = torch.sigmoid(attribute_output[0])
    preds = (probs >= 0.5).float()
    pred = preds.tolist()
    pred_str = [str(int(b)) for b in pred]
    return pred_str
def task_headpose(headpose_output):
    pitch = headpose_output[0][0].item() * 180 / np.pi
    pitch = round(pitch, 1)
    yaw = headpose_output[0][1].item() * 180 / np.pi
    yaw = round(yaw, 1)
    roll = headpose_output[0][2].item() * 180 / np.pi
    roll = round(roll, 1)

    raw=  headpose_output[0].detach().numpy().tolist()

    headpose_dict = {"pitch": pitch,
                     "yaw": yaw,
                     "roll": roll }

    return headpose_dict

def task_faceparsing(seg_output):
    preds = seg_output.softmax(dim=1)
    mask = torch.argmax(preds, dim=1)
    pred_mask = mask[0].detach().cpu().numpy()


    return pred_mask


def process_landmarks(landmark_output ):
    denorm_landmarks = denorm_points(landmark_output.view(-1, 68, 2)[0], 224, 224)
    denorm_landmarks=denorm_landmarks.detach().cpu()
   # image = unnormalize(images[0].detach().cpu())

    landmarks_list = []
    for landmark in denorm_landmarks[0]:
        x, y = landmark[0], landmark[1]
        # landmarks_list.append((int(x.item()), int(y.item())))
        landmarks_list.append((int(round(x.item())), int(round(y.item()))))

    # im = visualize_landmarks(image, denorm_landmarks, (255, 255, 0))
    # save_path_viz = os.path.join(args.results_path, "landmarks.png")
    # cv2.imwrite("./landmarks.png", im[:, :, ::-1])

    return landmarks_list

# def process_landmarks(landmark_output,images ):
#     denorm_landmarks = denorm_points(landmark_output.view(-1, 68, 2)[0], 224, 224)
#     denorm_landmarks=denorm_landmarks.detach().cpu()
#     image = unnormalize(images[0].detach().cpu())
#
#     landmarks_list = []
#     for landmark in denorm_landmarks[0]:
#         x, y = landmark[0], landmark[1]
#         # landmarks_list.append((int(x.item()), int(y.item())))
#         landmarks_list.append((int(round(x.item())), int(round(y.item()))))
#
#     # im = visualize_landmarks(image, denorm_landmarks, (255, 255, 0))
#     # save_path_viz = os.path.join(args.results_path, "landmarks.png")
#     # cv2.imwrite("./landmarks.png", im[:, :, ::-1])
#
#     return landmarks_list