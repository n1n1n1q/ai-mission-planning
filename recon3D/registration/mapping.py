"""
Mapping 2D objects to 3D points in a point cloud
"""
import numpy as np
import open3d as o3d
from collections import defaultdict
from recon3D.reconstruction.model import to_numpy

def map_classes(detector, output_dict, res_pcd, confidence=50):
    """
    Map class names to indices.
    """
    detected_pcds = defaultdict(list)
    detected_pcds_colors = defaultdict(list)
    pcds_frames = defaultdict(list)
    keep_frac = 1.0 - confidence / 100.0
    
    for i in range(len(output_dict["views"])):
        view = output_dict["views"][i]
        pred = output_dict["preds"][i]
        res = detector.process_frame_with_tracking(view["img"])
        pred["pts3d_local_aligned_to_global"] = pred["pts3d_local_aligned_to_global"].squeeze(0)
        view["img"] = view["img"].squeeze(0)
        if res is not None:
            for box in res:
                cls_id, x_min, y_min, x_max, y_max, conf, _, class_name = box
                pcds_frames[class_name].append(i)
                key = f"{class_name}_{cls_id}"
                block = pred["pts3d_local_aligned_to_global"][
                    int(y_min):int(y_max),
                    int(x_min):int(x_max)
                ]
                
                block_reshaped = block.reshape(-1, 3)
                if 'conf' in pred:
                    conf_block = pred["conf"].squeeze(0)[
                        int(y_min):int(y_max),
                        int(x_min):int(x_max)
                    ]
                    conf_flat = to_numpy(conf_block.cpu()).flatten()
                    
                    k = max(1, int(len(conf_flat) * keep_frac))
                    idx = np.argpartition(-conf_flat, k - 1)[:k]
                    
                    filtered_block = block_reshaped[idx]
                    detected_pcds[key].extend(filtered_block.tolist())
                    
                    color_block = view["img"][:, int(y_min):int(y_max), 
                                             int(x_min):int(x_max)].permute(1, 2, 0)
                    color_reshaped = color_block.reshape(-1, 3)
                    filtered_colors = color_reshaped[idx]
                    detected_pcds_colors[key].extend(filtered_colors.tolist())
                else:
                    detected_pcds[key].extend(block_reshaped.tolist())
                    color_block = view["img"][:, int(y_min):int(y_max), 
                                             int(x_min):int(x_max)].permute(1, 2, 0)
                    detected_pcds_colors[key].extend(color_block.reshape(-1, 3).tolist())
                    
    interesting_clouds = {k:helper(detected_pcds[k], detected_pcds_colors[k])  for k in detected_pcds.keys()}
    res_pcd.interesting_clouds = interesting_clouds
    res_pcd.obj_frames = pcds_frames

def helper(pts, clr):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.array(clr))
    return pcd