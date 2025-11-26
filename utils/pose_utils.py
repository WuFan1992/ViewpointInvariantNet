import numpy as np
import torch


def normalize(x):
    return x / np.linalg.norm(x)

"""
Full projection Function
"""
def ndc2pixel(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def fullproj(point_3d, full_proj_matrix, W, H):
    """
    Project the 3D point cloud into pixel space
    Using the 3DGS projection methods: World_coord --> Camera_coord 
    --> NDC_coord--> Pixel_coord 
    
    """
    hom = torch.matmul(point_3d, full_proj_matrix)
    weight = 1.0/(hom[:,3] + 0.000001)
    return ndc2pixel(hom[:,0]*weight, W), ndc2pixel(hom[:,1]*weight, H)

def project_w2p(points_3d, P, W, H):
    """
    Project the 3D points into 2D pixel spaces with given projection matrix
    The pixel that is projected out of the pixel space [0:W, 0:H] will be rejected
    
    Input:  
         points_3d: [N, 3] Tensor
         P: [3, 4] Projection Matrix Tensor

    Return :
         result: [M, 5] Tensor, Each line  [x, y, X, Y, Z] contains its pixel coordinates (x, y) and its asscociated 
                 3D point cloud coordinates (X, Y , Z)   
    """   
    N = points_3d.shape[0]
    
    # Construct homogeneous coordinates [X, Y, Z, 1]
    ones = torch.ones((N, 1), dtype=points_3d.dtype, device=points_3d.device)
    points_homogeneous = torch.cat([points_3d, ones], dim=1)  # [N, 4]
    
    # Project 3D points into pixel space
    P = P.to(points_3d.device) 
    x,y = fullproj(points_homogeneous, P, W, H) # 1920 1080 for cambridge
    
    # Keep only the projected pixel that is inside the pixel space ：x ∈ (0, 640), y ∈ (0, 480)
    mask = (x > 0) & (x < W) & (y > 0) & (y < H)
    
    x_filtered = x[mask]
    y_filtered = y[mask]
    points_3d_filtered = points_3d[mask]  # [M, 3]

    # Concetenate to [M, 5]： [x, y, X, Y, Z]
    result = torch.cat([x_filtered.unsqueeze(1), 
                        y_filtered.unsqueeze(1), 
                        points_3d_filtered], dim=1)   
    
    return mask,result




"""
Use to get the gt 3d coord from query keypoint
"""

def new_calculate_ndc2camera(proj_matrix, xndc, yndc, depth):
    a1 = proj_matrix[0,0]
    a2 = proj_matrix[0,1]
    a3 = proj_matrix[0,2]
    a4 = proj_matrix[0,3]
    
    a5 = proj_matrix[1,0]
    a6 = proj_matrix[1,1]
    a7 = proj_matrix[1,2]
    a8 = proj_matrix[1,3]
    
    
    a13 = proj_matrix[3,0]
    a14 = proj_matrix[3,1]
    a15 = proj_matrix[3,2]
    a16 = proj_matrix[3,3]
    
    A1 = a1-xndc*a13
    B1 = a2-xndc*a14
    C1 = (a3-xndc*a15)*depth+a4-xndc*a16
    
    A2 = a5-yndc*a13
    B2 = a6-yndc*a14
    C2 = (a7-yndc*a15)*depth+a8-yndc*a16
    
    X = (-C1*B2+C2*B1)/(A1*B2-A2*B1)
    Y = (-A1*C2+A2*C1)/(A1*B2-A2*B1)
    
    return X, Y

def pixel2ndc(pixel, S):
    return (((pixel/0.5)+1.0)/S)-1.0

def getGTXYZ(camera2ndc, view2camera, point_2d, depth_map):
    #Get the depth value
    depth_map = depth_map.detach().squeeze(0)
    depth = depth_map[point_2d[:,1].int().to("cpu"), point_2d[:,0].int().to("cpu")] 
    X, Y = new_calculate_ndc2camera(camera2ndc.transpose(0,1), pixel2ndc(point_2d[:,0], 640), pixel2ndc(point_2d[:,1], 480), depth)
    ones = torch.tensor([1.0]).repeat(point_2d.size(0)).to("cuda")
        
    cam_coord_inv = torch.stack([X, Y, depth, ones], dim=1)
    output = torch.matmul(cam_coord_inv.double(), torch.inverse(view2camera).double())
    return output[:, :3]