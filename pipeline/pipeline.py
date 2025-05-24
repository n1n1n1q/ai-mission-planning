import os
import logging

from pathlib import Path

# from recon3D.data.video_splitter import video_to_frames
# from recon3D.data.utils import visualize_pcds
# from recon3D.reconstruction.model import load_images, inference, merge_clouds

from recon3D.data.io import save_output_dict, load_output_dict

import torch


def video_to_point_cloud(video_path):
    frames_output_path = f"data/photos/frames_of_{Path(video_path).stem}"
    time_intervals, num_of_frames = video_to_frames(
        video_path, frames_output_path, frames_per_second=1
    )

    images = load_images(frames_output_path)
    logging.info(f"Loaded {len(images)} images from {frames_output_path}")
    output_dict = inference(images)
    logging.info("Inference completed")
    pcd = merge_clouds(output_dict)

    return pcd, time_intervals, num_of_frames



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
