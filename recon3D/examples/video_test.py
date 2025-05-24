"""
Fast3R test script
"""
import logging
from pathlib import Path
from recon3D.reconstruction.model import *
from recon3D.data.utils import *
from recon3D.data.io import save_output_dict, load_output_dict
from recon3D.data.video_splitter import video_to_frames

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    filepath = "data/IMG_2265.MOV"
    frames_output_path = f"data/photos/"
    video_to_frames(filepath, frames_output_path, frames_per_second=1)

    images = load_data(frames_output_path)
    logging.info(f"Loaded {len(images)} images from {filepath}")
    output_dict = inference(images)
    save_output_dict(output_dict, "saved_reconstruction.pkl")
    logging.info("Inference completed")
    pcd = merge_clouds(output_dict, confidence=20)
    visualize_pcds(pcd.pcd)

