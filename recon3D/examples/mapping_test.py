from recon3D.object_detection.detector import Detector
import logging
from recon3D.reconstruction.model import *
from recon3D.data.utils import *
from collections import defaultdict
from recon3D.registration.mapping import map_classes

logging.basicConfig(level=logging.INFO)

# def map_classes(detector, output_dict, res_pcd):
#     """
#     Map class names to indices.
#     """
#     detected_pcds = defaultdict(list)
#     detected_pcds_colors = defaultdict(list)
#     pcds_frames = defaultdict(list)
#     for i in range(len(output_dict["views"])):
#         view = output_dict["views"][i]
#         pred = output_dict["preds"][i]
#         res = detector.process_frame_with_tracking(view["img"])
#         pred["pts3d_local_aligned_to_global"] = pred["pts3d_local_aligned_to_global"].squeeze(0)
#         view["img"] = view["img"].squeeze(0)
#         if res is not None:
#             for box in res:
#                 cls_id, x_min, y_min, x_max, y_max, conf, _, class_name = box
#                 pcds_frames[class_name].append(i)
#                 key = f"{class_name}{cls_id}"
#                 block = pred["pts3d_local_aligned_to_global"][
#                     int(y_min):int(y_max),
#                     int(x_min):int(x_max)
#                 ]
#                 detected_pcds[key].extend(block.reshape(-1, 3).tolist())
#                 color_block = view["img"][:, int(y_min):int(y_max), 
#                                           int(x_min):int(x_max)].permute(1, 2, 0)
#                 detected_pcds_colors[key].extend(color_block.reshape(-1, 3).tolist())
#     def helper(pts, clr):
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(np.array(pts))
#         pcd.colors = o3d.utility.Vector3dVector(np.array(clr))
#         return pcd

#     interesting_clouds = {k:helper(detected_pcds[k], detected_pcds_colors[k])  for k in detected_pcds.keys()}
#     res_pcd.interesting_clouds = interesting_clouds
#     res_pcd.obj_frames = pcds_frames

if __name__ == "__main__":
    filepath = "data/photos"
    detector = Detector()
    images = load_data(filepath)
    logging.info(f"Loaded {len(images)} images from {filepath}")
    output_dict = inference(images)
    logging.info("Inference completed")
    global_pcd = merge_clouds(output_dict)
    map_classes(detector, output_dict, global_pcd)
    for class_name, pcd in global_pcd.interesting_clouds.items():
        visualize_pcds(pcd)