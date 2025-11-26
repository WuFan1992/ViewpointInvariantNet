import torch
from torch import nn, optim
import numpy as np
import os
from utils.graphics_utils import BasicPointCloud
import torch.nn.functional as F
import e3nn.o3 as o3
from utils.system_utils import mkdir_p

from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_points3D_binary, read_points3D_text


def storePly(path, xyz, rgb, ids_3d):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('id_3d', 'int32')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, ids_3d), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    ids_3d = vertices['id_3d']
    return BasicPointCloud(points=positions, colors=colors, normals=normals, ids_3d=ids_3d)


class MLP(nn.Module):
     def __init__(self, input_channels: int = 16, output_channels : int = 64):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, output_channels)
        )
     
     def forward(self, input: torch.Tensor):
         return self.mlp(input)
             



class SHPointCloud(nn.Module):
    def __init__(self, mlp: nn.Module, device="cuda", lmax: int = 3, num_channels: int = 16, out_dim: int = 64):
        """
        coords: [N, 3] Tensor
        init_sh_features: [N, 64] Tensor or None
        """
        super().__init__()
      
        self.device = device
        self.lmax = lmax
        self.sh_dim = (lmax+1)**2
        self.num_channels = num_channels
        self.out_dim = out_dim
        
        self.mlp = mlp
    
    def getMLP(self):
        return self.mlp
    
    def get_xyz(self):
        return self._xyz
    
    def get_shcoeffs(self):
        return self.sh_coeffs
        
                
    def get_valid_indices(self, point_ids):
        """
        Pure PyTorch version of get_valid_indices, no numpy involved.
        Returns: (M,) LongTensor of indices into self._xyz and self.sh_coeffs
        """
        valid_mask = point_ids != -1                     # [B]
        filtered_ids = point_ids[valid_mask]             # [M], still on CUDA

        # Build a map from global_id → index
        # self._ids_3d is shape (N,), contains global IDs
        # We use broadcasting to compare all pairs
        # (M, 1) == (1, N) → (M, N) boolean
        matches = filtered_ids[:, None] == self._ids_3d[None, :]  # [M, N]
    
        # Get index of the first match per row
        valid_indices = matches.float().argmax(dim=1)  # [M]
        return valid_indices
        
    def forward(self, point_ids, cam_center):
        """
            point_ids: (B,) LongTensor with possibly -1 values
            cam_center: (3,) Tensor on the same device
            Returns:
                (M, out_dim) Tensor features for valid points
        """
        point_ids = torch.from_numpy(point_ids).long().to(self.device)
        # Get valid indices purely on torch tensors
        valid_indices = self.get_valid_indices(point_ids)  # (M,)

        # Use valid indices to index self._xyz and self.sh_coeffs
        cam_center = torch.from_numpy(cam_center).float().to(self.device)
        directions = cam_center[None, :] - self._xyz[valid_indices]  # (M,3)
        directions = F.normalize(directions, dim=-1)                # (M,3)

        # Evaluate SH basis (M, sh_dim)
        Y = o3.spherical_harmonics(
        list(range(self.lmax + 1)),
        directions,
        normalize=True,
        normalization='component'
        ).float().to(self.device)  # Make sure on device

        # SH coefficients (M, num_channels, sh_dim)
        sh_selected = self.sh_coeffs[valid_indices]  # (M, num_channels, sh_dim)
        confi_selected = self.confi[valid_indices]

        # SH dot product (M, num_channels)
        sh_feats = torch.einsum('mcd,md->mc', sh_selected, Y)  # (M, num_channels)
        confi_res = torch.einsum('mcd,md->mc', confi_selected, Y)  # (M, 1)

        # Project to output dimension
        out_feats = self.mlp(sh_feats)  # (M, out_dim)

        return out_feats, confi_res
    
    def init_sh_pc(self, source_path):
        """
        fetch the point cloud 
        """
        # Generate the ply file
        self.generate_ply(source_path)
        
        # Get the point cloud path
        pc_path = os.path.join(source_path,"sparse/0/points3D_withidx.ply")
        # Load the pcd
        pcd = fetchPly(pc_path)
        # set the initial xyz 
        self._xyz = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # initialize the feature
        self.num_points = self._xyz.shape[0]
        # Set the point cloud 3D
        self._ids_3d = torch.tensor(np.array(pcd.ids_3d), dtype=torch.long, device=self.device)
        self.sh_coeffs = nn.Parameter(
                torch.randn(self.num_points, self.num_channels, self.sh_dim, device=self.device) * 0.01)
        
        self.confi = nn.Parameter(
                torch.randn(self.num_points, 1 , self.sh_dim, device=self.device) * 0.01)
    
    
    def generate_ply(self, path):
        ply_path = os.path.join(path, "sparse/0/points3D_withidx.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _, ids_3d = read_points3D_binary(bin_path)
            
            except:
                xyz, rgb, _, ids_3d = read_points3D_text(txt_path)
            
            storePly(ply_path, xyz, rgb, ids_3d)
            
        
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self.sh_coeffs.shape[1]*self.sh_coeffs.shape[2]):  
            l.append('semantic_{}'.format(i))
        return l
        
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        semantic_feature = self.sh_coeffs.detach().flatten(start_dim=1).contiguous().cpu().numpy() 
        print("semantic feature shape = ", semantic_feature.shape)
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, semantic_feature), axis=1) 
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def load_ply(self, path):
        plydata = PlyData.read(path)

        self._xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        
        count = sum(1 for name in plydata.elements[0].data.dtype.names if name.startswith("semantic_"))
        semantic_feature = np.stack([np.asarray(plydata.elements[0][f"semantic_{i}"]) for i in range(count)], axis=1) 
        self.sh_coeffs = torch.from_numpy(np.expand_dims(semantic_feature, axis=-1)).view(-1,16,16) 
    

        
        
