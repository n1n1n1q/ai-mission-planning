import os
import logging

from pathlib import Path

from recon3D.data.video_splitter import video_to_frames
from recon3D.data.utils import visualize_pcds
from recon3D.reconstruction.model import load_images, inference, merge_clouds


def video_to_point_cloud(video_path):
    frames_output_path = f"data/photos/frames_of_{Path(video_path).stem}"
    video_to_frames(video_path, frames_output_path, frames_per_second=1)

    images = load_images(frames_output_path)
    logging.info(f"Loaded {len(images)} images from {frames_output_path}")
    output_dict = inference(images)
    logging.info("Inference completed")
    pcd = merge_clouds(output_dict)
    visualize_pcds(pcd)


if __name__ == "__main__":
    video_file = "assets/hackaton videos /IMG_2265.MOV"
    video_to_point_cloud(video_file)
