import os
import logging

from pathlib import Path

from recon3D.data.video_splitter import video_to_frames

from recon3D.data.utils import visualize_pcds, to_pointcloud
from recon3D.reconstruction.model import merge_clouds
from recon3D.object_detection.detector import Detector

from recon3D.data.io import save_output_dict, load_output_dict

from recon3D.registration.mapping import map_classes

from recon3D.registration.icp import icp

import torch


def video_to_point_cloud(video_path, pre_output_dict_path=None):
    frames_output_path = f"data/photos/frames_of_{Path(video_path).stem}"
    time_intervals, num_of_frames = video_to_frames(
        video_path, frames_output_path, frames_per_second=1
    )

    if pre_output_dict_path is None:
        images = load_images(frames_output_path)
        logging.info(f"Loaded {len(images)} images from {frames_output_path}")
        output_dict = inference(images)
        logging.info("Inference completed")
        pcd = merge_clouds(output_dict)

    else:
        output_dict = load_output_dict(pre_output_dict_path, torch.device("cpu"))
        pcd = merge_clouds(output_dict, confidence=20)

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

    if False:
        for class_name, pcs in pcd1.interesting_clouds.items():
            visualize_pcds(pcs, window_name=class_name)

        for class_name, pcs in pcd1.interesting_clouds.items():
            visualize_pcds(pcs, window_name="second" + class_name)

    day1_to_day2 = icp(pcd1.pcd, pcd2.pcd)
    print(day1_to_day2)


if __name__ == "__main__":
    video_file1 = "assets/hackaton videos /IMG_2265.MOV"
    reconstruction_file_path1 = "data/saved_reconstruction_day1.pkl"

    video_file2 = "assets/hackaton videos /IMG_2266.MOV"
    reconstruction_file_path2 = "data/saved_reconstruction_day2.pkl"

    load_videos_map_objects(
        video_file1, video_file2, reconstruction_file_path1, reconstruction_file_path2
    )
