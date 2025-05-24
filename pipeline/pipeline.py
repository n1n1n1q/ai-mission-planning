import os
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from recon3D.data.video_splitter import video_to_frames
from recon3D.data.utils import visualize_pcds
from recon3D.reconstruction.model import load_data, inference, merge_clouds
from recon3D.data.io import save_output_dict, load_output_dict
from recon3D.pipeline.object_reconstruction_pipeline import process_video_with_objects

import torch


def video_to_point_cloud(video_path):
    frames_output_path = f"data/photos/frames_of_{Path(video_path).stem}"
    time_intervals, num_of_frames = video_to_frames(
        video_path, frames_output_path, frames_per_second=1
    )

    images = load_data(frames_output_path)
    logging.info(f"Loaded {len(images)} images from {frames_output_path}")
    output_dict = inference(images)
    logging.info("Inference completed")
    pcd = merge_clouds(output_dict)

    return pcd, time_intervals, num_of_frames


def video_to_point_cloud_with_objects(video_path, output_dir=None, target_classes=None):
    """
    Enhanced function for video to point cloud conversion with object detection.
    Ensures unique objects are stored only once in CloudWithViews.interest_clouds.

    Args:
        video_path: Path to input video
        output_dir: Directory to save intermediate results
        target_classes: List of object classes to detect (None for all)

    Returns:
        CloudWithViews with unique objects in interest_clouds
    """
    return process_video_with_objects(
        video_path=video_path,
        output_dir=output_dir,
        confidence_threshold=0.3,
        reconstruction_confidence=65,
        target_classes=target_classes,
        frames_per_second=1,
    )


if __name__ == "__main__":
    video_file = "assets/hackaton videos /IMG_2265.MOV"
    reconstruction_file_path = "data/saved_reconstruction.pkl"

    output_dict = load_output_dict(reconstruction_file_path, torch.device("cpu"))
    preds_0 = output_dict["preds"][0]
    print(preds_0.keys())
    print(preds_0["pts3d_in_other_view"].shape)
    print(preds_0["pts3d_local"].shape)
    print(preds_0["conf_local"].shape)
    print(preds_0["conf"].shape)

    # video_to_point_cloud(video_file)
