import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
import cv2



import sys
from random import randint
from scene import Scene
from argparse import ArgumentParser
from tqdm import tqdm
import os
import uuid

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


from Net.Pose.poseregression import PoseRegressorMLP
from arguments import ModelParams
from PIL import Image

from Net.XFeat.modules.xfeat import XFeat


from utils.general_utils import image_process

from Net.Pose.posenetmlp import PoseNetMLP

# -----------------------------
# 1) PoseNet Model
# -----------------------------
class PoseNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 1) Backbone CNN (ResNet18)
        backbone = models.resnet18(pretrained=pretrained)
        modules = list(backbone.children())[:-2]  # 去掉 avgpool + fc
        self.feature_extractor = nn.Sequential(*modules)  # 输出 (B, 512, H/32, W/32)

        # 2) Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> (B, 512, 1, 1)

        # 3) MLP for pose regression
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc_translation = nn.Linear(1024, 3)
        self.fc_rotation = nn.Linear(1024, 4)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.feature_extractor(x)          # -> (B, 512, H/32, W/32)
        x = self.global_pool(x)                # -> (B, 512, 1, 1)
        x = x.view(x.size(0), -1)              # -> (B, 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        t = self.fc_translation(x)
        q = self.fc_rotation(x)
        q = q / torch.norm(q, dim=1, keepdim=True)  # Normalize quaternion
        return t, q

# -----------------------------
# 2) PoseNet loss
# -----------------------------
def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为3x3旋转矩阵
    q: (..., 4) tensor，四元数格式 (x, y, z, w)
    返回: (..., 3, 3) tensor
    """
    # 确保四元数归一化
    q = q / q.norm(dim=-1, keepdim=True)
    
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # 计算旋转矩阵元素
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot = torch.zeros(q.shape[:-1] + (3, 3), device=q.device, dtype=q.dtype)

    rot[..., 0, 0] = 1 - 2*(yy + zz)
    rot[..., 0, 1] = 2*(xy - wz)
    rot[..., 0, 2] = 2*(xz + wy)

    rot[..., 1, 0] = 2*(xy + wz)
    rot[..., 1, 1] = 1 - 2*(xx + zz)
    rot[..., 1, 2] = 2*(yz - wx)

    rot[..., 2, 0] = 2*(xz - wy)
    rot[..., 2, 1] = 2*(yz + wx)
    rot[..., 2, 2] = 1 - 2*(xx + yy)

    return rot



def training_report(tb_writer, loss, iteration):
     if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', loss.item(), iteration)
        
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def calculate_pose_errors(R_gt, t_gt, R_est, t_est):
    # Calculate rotation error
    rotError = np.matmul(R_est.T, R_gt)
    rotError = cv2.Rodrigues(rotError)[0]
    rotError = np.linalg.norm(rotError) * 180 / np.pi

    # Calculate translation error
    transError = np.linalg.norm(t_gt - t_est) * 100  # Convert to cm
    #transError = np.linalg.norm(-R_gt.T @ t_gt + R_est.T @ t_est, axis=0)*100
    #transError = np.median(transError)
    
    return rotError, transError



    
def validation(dataset: ModelParams):
        
    rot_err_list = []
    trans_err_list = []
    tb_writer = prepare_output_and_logger(dataset)
    
    scene = Scene(dataset, load_iteration=15000)
    model = PoseNet(pretrained=False).to(args.data_device)
    model.load_state_dict(torch.load("./weights/pretrain_poseregression.pth", weights_only=True))
        
    views_test = scene.getTestCameras().copy()

    
    for _, view in enumerate(tqdm(views_test, desc="Matching progress")):
                
        try:
            image = Image.open(view.image_path) 
        except:
            print(f"Error opening image: {view.image_path}")
            continue
        
        original_image = image_process(image)

        gt_image = original_image.unsqueeze(0).to(args.data_device)
        t_pred, q_pred = model(gt_image)

        q_pred = quaternion_to_rotation_matrix(q_pred)
        t_pred = t_pred.detach().cpu().squeeze(0).numpy()
        q_pred = q_pred.detach().cpu().squeeze(0).numpy() 
       
        
        gt_R, gt_t = view.R, view.T
        

        rotError, transError = calculate_pose_errors(gt_R, gt_t, q_pred.T, t_pred)

        rot_err_list.append(rotError)
        trans_err_list.append(transError)
        
    err_mean_rot =  np.mean(rot_err_list)
    err_mean_trans = np.mean(trans_err_list)
    print(f"Rotation Average Error: {err_mean_rot} deg ")
    print(f"Translation Average Error: {err_mean_trans} cm ")        
        
    
    

    
if __name__ == "__main__":
# Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)

    args = parser.parse_args(sys.argv[1:])
    
    args.eval = True
    # Initialize system state (RNG)
    validation(model.extract(args))