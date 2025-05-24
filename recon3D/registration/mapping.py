"""
Mapping 2D objects to 3D points in a point cloud
"""
import numpy as np
import open3d as o3d
from collections import defaultdict


def map_classes(detector, output_dict, res_pcd, segmentation=False):
    """
    Map class names to indices.
    """
    detected_pcds = defaultdict(list)
    detected_pcds_colors = defaultdict(list)
    pcds_frames = defaultdict(list)
    for i in range(len(output_dict["views"])):
        view = output_dict["views"][i]
        pred = output_dict["preds"][i]
        if segmentation:
            res = detector.process_frame_with_tracking_and_segmentation(view["img"])
        else:
            res = detector.process_frame_with_tracking(view["img"])
        pred["pts3d_local_aligned_to_global"] = pred["pts3d_local_aligned_to_global"].squeeze(0)
        view["img"] = view["img"].squeeze(0)
        if res is not None:
            for box in res:
                if segmentation:
                    cls_id, x_min, y_min, x_max, y_max, _, _, class_name, mask = box
                else:
                    cls_id, x_min, y_min, x_max, y_max, _, _, class_name = box
                pcds_frames[class_name].append(i)
                key = f"{class_name}{cls_id}"
                if segmentation:
                    mask = mask.astype(bool)
                    block = pred["pts3d_local_aligned_to_global"][
                        mask, :
                    ]
                    detected_pcds[key].extend(block.reshape(-1, 3).tolist())
                    flat_colors = view["img"].reshape(3, -1).T
                    flat_mask = mask.ravel().astype(bool)
                    color_block = flat_colors[flat_mask]
                    detected_pcds_colors[key].extend(color_block.reshape(-1, 3).tolist())
                else:
                    block = pred["pts3d_local_aligned_to_global"][
                        int(y_min):int(y_max),
                        int(x_min):int(x_max)
                    ]
                    detected_pcds[key].extend(block.reshape(-1, 3).tolist())
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
