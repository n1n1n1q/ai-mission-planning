"""
Reconstruction with Fast3R
"""

import os
import torch
import open3d as o3d
import numpy as np
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference as fast3r_inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from recon3D.data.cloud import CloudWithViews

model = Fast3R.from_pretrained("model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
model.eval()
lit_module.eval()


def load_data(filepath):
    """
    Load data from a file path.
    This function should be implemented to load your specific data format.
    """
    filelist = os.listdir(filepath)
    filelist = [
        os.path.join(filepath, f)
        for f in filelist
        if f.endswith(".jpg") or f.endswith(".png")
    ]
    images = load_images(filelist, size=512, verbose=True)
    return images


def inference(images):
    """
    Perform inference on the images using the Fast3R model.
    """
    output_dict, profiling_info = fast3r_inference(
        images,
        model,
        device,
        dtype=torch.float32,
        verbose=True,
        profiling=True,
    )
    return output_dict


def extract_point_clouds(output_dict):
    """
    Extract point clouds from the output dictionary.
    """
    point_clouds = []
    for pred in output_dict["preds"]:
        point_cloud = pred["pts3d_in_other_view"].cpu().numpy()
        point_clouds.append(point_cloud)
    return point_clouds


def extract_poses(output_dict):
    """
    Extract camera positions from the output dictionary.
    """
    poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict["preds"],
        niter_PnP=100,
        focal_length_estimation_method="first_view_from_global_head",
    )
    camera_poses = poses_c2w_batch[0]
    return camera_poses

def estimate_global_poses(output_dict, confidence=65):
    """
    Estimate global point cloud positions.
    """
    lit_module.align_local_pts3d_to_global(
        preds=output_dict["preds"],
        views=output_dict["views"],
        min_conf_thr_percentile=confidence,
    )

def merge_clouds(output_dict, confidence=65):
    """
    Merge point clouds into a single point cloud.
    """
    estimate_global_poses(output_dict, confidence=confidence)
    top_points, top_colors = [], []
    keep_frac = 1.0 - confidence / 100.0
    for pred, view in zip(output_dict["preds"], output_dict["views"]):
        pts = to_numpy(pred["pts3d_in_other_view"].cpu()).reshape(-1, 3)
        conf = to_numpy(pred["conf"].cpu()).flatten()

        k = max(1, int(len(conf) * keep_frac))
        idx = np.argpartition(-conf, k - 1)[:k]
        pts, conf = pts[idx], conf[idx]

        clr = to_numpy(view["img"].cpu().squeeze().permute(1, 2, 0)).reshape(-1, 3)[idx]
        clr = ((clr + 1.0) * 127.5).astype(np.uint8) / 255.0  # → [0,1]

        top_points.append(pts)
        top_colors.append(clr)

    points = np.concatenate(top_points, axis=0)
    colors = np.concatenate(top_colors, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return CloudWithViews(pcd=pcd, poses=extract_poses(output_dict), views=output_dict["views"])


def to_numpy(torch_tensor):
    """
    Convert a PyTorch tensor to a NumPy array.
    """
    return torch_tensor.cpu().numpy() if torch_tensor.is_cuda else torch_tensor.numpy()
