import os
import logging
import numpy as np
import copy
import open3d as o3d

from pathlib import Path

from recon3D.data.video_splitter import video_to_frames
from recon3D.data.utils import visualize_pcds, to_pointcloud
from recon3D.reconstruction.model import load_data, inference, merge_clouds
from recon3D.object_detection.detector import Detector
from recon3D.data.io import save_output_dict, load_output_dict
from recon3D.registration.mapping import map_classes
from recon3D.registration.icp import icp
from recon3D.data.cloud import compare_objects_hausdorff

import torch


def video_to_point_cloud(video_path, pre_output_dict_path=None):
    frames_output_path = f"data/photos/frames_of_{Path(video_path).stem}"
    time_intervals, num_of_frames = video_to_frames(
        video_path, frames_output_path, frames_per_second=1
    )

    if pre_output_dict_path is None:
        images = load_data(frames_output_path)
        logging.info(f"Loaded {len(images)} images from {frames_output_path}")
        output_dict = inference(images)
        logging.info("Inference completed")
        pcd = merge_clouds(output_dict)

    else:
        output_dict = load_output_dict(pre_output_dict_path, torch.device("cpu"))
        pcd = merge_clouds(output_dict, confidence=45)

    return output_dict, pcd, time_intervals, num_of_frames


def load_videos_map_objects(
    video1, video2, pre_output_dict1=None, pre_output_dict2=None
):
    output_dict1, pcd1, time_intervals1, num_of_frames1 = video_to_point_cloud(
        video1, pre_output_dict1
    )

    output_dict2, pcd2, time_intervals2, num_of_frames2 = video_to_point_cloud(
        video2, pre_output_dict2
    )

    detector = Detector()

    map_classes(detector, output_dict1, pcd1)
    map_classes(detector, output_dict2, pcd2)

    print(pcd1.obj_frames)
    print(pcd1.interesting_clouds)

    if False:
        for class_name, pcs in pcd1.interesting_clouds.items():
            visualize_pcds(pcs, window_name=class_name)

        for class_name, pcs in pcd1.interesting_clouds.items():
            visualize_pcds(pcs, window_name="second" + class_name)

    if True:
        if pcd1.interesting_clouds is not None:
            print(pcd1.interesting_clouds.keys())
        else:
            print("pcd1.interesting_clouds is None")

        if pcd2.interesting_clouds is not None:
            print(pcd2.interesting_clouds.keys())
        else:
            print("pcd2.interesting_clouds is None")

        pcd1.pcd.paint_uniform_color([1.0, 0.0, 0.0])
        pcd2.pcd.paint_uniform_color([0.0, 1.0, 0.0])

    day1_to_day2 = icp(pcd1.pcd, pcd2.pcd)

    pcd1 = pcd1 @ day1_to_day2

    visualize_pcds(pcd1.pcd, pcd2.pcd)

    # Compare objects using Hausdorff distance
    missing_obj_frames12 = compare_objects_hausdorff(pcd1, pcd2, threshold=0.7)

    print(
        f"Found {len(missing_obj_frames12)} objects in day1 that are missing or significantly different in day2"
    )

    missing_obj_frames21 = compare_objects_hausdorff(pcd2, pcd1, threshold=0.7)

    print(
        f"Found {len(missing_obj_frames21)} objects in day2 that are missing or significantly different in day1"
    )

    # Optional: visualize comparisons (keeping the original visualization if needed)
    if False:
        for class_name_1, obj1 in pcd1.interesting_clouds.items():
            for class_name_2, obj2 in pcd2.interesting_clouds.items():

                obj1.paint_uniform_color([1.0, 0.0, 0.0])
                obj2.paint_uniform_color([0.0, 1.0, 0.0])
                visualize_pcds(
                    obj1, obj2, window_name=f"{class_name_1} to {class_name_2}"
                )

    for i, missing_obj in enumerate(missing_obj_frames12):
        message = f"Missing object {i+1} on day 1: {missing_obj['obj_name']} -> {missing_obj['obj_frame']}"
        print(message)
        pcd1.interesting_clouds[missing_obj["obj_name"]].paint_uniform_color(
            [1.0, 0.0, 0.0]
        )
        visualize_pcds(
            pcd1.pcd,
            pcd1.interesting_clouds[missing_obj["obj_name"]],
            window_name=message,
        )

    for i, missing_obj in enumerate(missing_obj_frames21):
        message = f"Missing object {i+1} on day 2: {missing_obj['obj_name']} -> {missing_obj['obj_frame']}"
        print(message)
        pcd2.interesting_clouds[missing_obj["obj_name"]].paint_uniform_color(
            [0.0, 1.0, 0.0]
        )
        visualize_pcds(
            pcd2.pcd,
            pcd2.interesting_clouds[missing_obj["obj_name"]],
            window_name=message,
        )

    return missing_obj_frames12, missing_obj_frames21


if __name__ == "__main__":
    video_file1 = "assets/hackaton videos /IMG_2265.MOV"
    reconstruction_file_path1 = "data/saved_reconstruction_day1.pkl"

    video_file2 = "assets/hackaton videos /IMG_2266.MOV"
    reconstruction_file_path2 = "data/saved_reconstruction_day2.pkl"

    day1_missing_in_day2, day2_missing_in_day1 = load_videos_map_objects(
        video_file1, video_file2, reconstruction_file_path1, reconstruction_file_path2
    )

    for i, missing_obj in enumerate(day1_missing_in_day2):
        print(
            f"Missing object {i+1} on day 1: {missing_obj['obj_name']} -> {missing_obj['obj_frame']}"
        )
