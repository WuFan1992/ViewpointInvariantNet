import torch
import sys
from random import randint
from scene import Scene
from argparse import ArgumentParser
from tqdm import tqdm
import cv2
import torch.nn.functional as F
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
from Net.invariant_feat_net import InvariantFeatureNet

from utils.general_utils import image_process

from Net.xfeat_pose_net import XFeatPoseNet



def rotation_matrix_to_quaternion(R):
    """
    R: (B, 3, 3) float32
    return q: (B, 4) float32 quaternion (x, y, z, w)
    """
    R = R.to(dtype=torch.float32)
    B = R.shape[0]

    q = torch.zeros((B, 4), device=R.device, dtype=torch.float32)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0
    mask = trace > 0
    if mask.any():
        t = trace[mask]
        s = torch.sqrt(t + 1.0) * 2  # ← float32
        q[mask, 3] = 0.25 * s
        q[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
        q[mask, 1] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
        q[mask, 2] = (R[mask, 1, 0] - R[mask, 0, 1]) / s

    # Case 2: R00 is largest
    mask2 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        t = 1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]
        s = torch.sqrt(t + 1.0) * 2
        q[mask2, 0] = 0.25 * s
        q[mask2, 3] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        q[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 2] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    # Case 3: R11 is largest
    mask3 = (~mask) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        t = 1.0 - R[mask3, 0, 0] + R[mask3, 1, 1] - R[mask3, 2, 2]
        s = torch.sqrt(t + 1.0) * 2
        q[mask3, 1] = 0.25 * s
        q[mask3, 3] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        q[mask3, 0] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        q[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    # Case 4: R22 is largest
    mask4 = ~(mask | mask2 | mask3)
    if mask4.any():
        t = 1.0 - R[mask4, 0, 0] - R[mask4, 1, 1] + R[mask4, 2, 2]
        s = torch.sqrt(t + 1.0) * 2
        q[mask4, 2] = 0.25 * s
        q[mask4, 3] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        q[mask4, 0] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        q[mask4, 1] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s

    # Normalize
    q = q / torch.norm(q, dim=1, keepdim=True)

    return q



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

def pose_loss(t_pred, q_pred, t_gt, q_gt, beta=250):
    """
    translation: (B, 3)
    rotation: (B, 3, 3)
    t_gt: (B, 3)
    R_gt: (B, 3, 3)
    """

   # 平移损失
    loss_t = torch.norm(t_pred - t_gt, dim=1).mean()

    # 单位化四元数
    q_pred = q_pred / torch.norm(q_pred, dim=1, keepdim=True)
    q_gt = q_gt / torch.norm(q_gt, dim=1, keepdim=True)

    # 旋转损失: quaternion distance
    # (等价于 rotation angle)
    dot = torch.sum(q_pred * q_gt, dim=1)
    loss_r = 1 - dot**2           # 更平滑
    loss_r = loss_r.mean()

    # 总损失（PoseNet 的标准做法）
    loss = loss_t + beta * loss_r

    return loss, loss_t, loss_r



def pretraining(dataset: ModelParams):
        
    tb_writer = prepare_output_and_logger(dataset)
    
    scene = Scene(dataset, load_iteration=15000)
    
    xfeat = XFeat(top_k=4096)
    
    invariant_net = InvariantFeatureNet(xfeat).to(args.data_device) 
    xfeat_posenet = XFeatPoseNet(feat_dim=64).to(args.data_device)
    xfeat_posenet.load_state_dict(torch.load("./weights/pretrain_poseregression.pth", weights_only=True, map_location=args.data_device))
        
    
    optimizer = torch.optim.Adam([
    {"params": invariant_net.parameters(), "lr": 1e-4},
    {"params": xfeat_posenet.parameters(), "lr": 5e-5},
])
    
    viewpoint_stack = scene.getTrainCameras().copy()

    num_iter = 15000
    
    progress_bar = tqdm(range(0, num_iter), desc="Training progress")

       
    for iteration in range(num_iter+1):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        
        try:
            image = Image.open(viewpoint_cam.image_path) 
        except:
            print(f"Error opening image: {viewpoint_cam.image_path}")
            continue
        
        original_image = image_process(image)
        
        gt_image = original_image.cuda()
        gt_feature_map = xfeat.get_descriptors(gt_image[None])
        #Regress Pose
        feat_map = gt_feature_map.clone()
        recon, invariant_map, adv_emd = invariant_net(feat_map)
        tran, rot = xfeat_posenet(adv_emd)
        
        
        gt_R = rotation_matrix_to_quaternion(torch.tensor(viewpoint_cam.R).unsqueeze(0))
        
        gt_R, gt_t = gt_R.to(args.data_device), torch.tensor(viewpoint_cam.T).to(args.data_device).unsqueeze(0)

        adv_loss, L_pos, L_rot = pose_loss(tran, rot, gt_t, gt_R)
        recon_loss = F.mse_loss(recon, feat_map)
        loss = recon_loss + 0.5*adv_loss
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(10)
            training_report(tb_writer, loss, iteration)
    progress_bar.close()
    
    torch.save(xfeat_posenet.state_dict(), "./weights/pretrain_poseregression.pth")

    

if __name__ == "__main__":
# Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)

    args = parser.parse_args(sys.argv[1:])
    
    args.eval = True
    # Initialize system state (RNG)
    pretraining(model.extract(args))

 

