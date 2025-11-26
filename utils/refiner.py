import torch
from utils.pose_utils import ndc2pixel
import torch.optim as optim




def project_w2c(points_3d, R, t):
    points_c = torch.matmul(points_3d, R) + t
    last = points_c.sum(dim=1, keepdim=True)    
    points_homo = torch.cat([points_c, last], dim=1)
    
    return points_homo

def project_c2p(points_3d, P, W, H):
    P = P.to(points_3d.device) 
    hom = torch.matmul(points_3d, P)
    
    weight = 1.0/(hom[:,3] + 0.000001)
    return ndc2pixel(hom[:,0]*weight, W), ndc2pixel(hom[:,1]*weight, H)


def refiner(view, query_kps, matched_3d, W, H):
        
    R_update = torch.tensor(view.R, require_grad=True)
    t_update = torch.tensor(view.t, require_grad=True)
    
    matrix_c2p = view.projection_matrix
    
    optimizer = torch.optim.Adam([R_update, t_update], lr=1e-2)
    
    for iter in range(500):
        optimizer.zero_grad()
        
        points_cam = project_w2c(query_kps, R_update, t_update)
        
        
        
        
     
    