"""
Fast3R test script
"""
import logging
from recon3D.reconstruction.model import *
from recon3D.data.utils import *
from recon3D.data.io import save_output_dict, load_output_dict
from recon3D.data.video_splitter import video_to_frames

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    video_path_1 = "data/day1.MOV"
    video_path_2 = "data/day2.MOV"
    video_to_frames(video_path_1, "data/day1_frames", frames_per_second=1)
    video_to_frames(video_path_2, "data/day2_frames", frames_per_second=1)
    images1 = load_data("data/day1_frames")
    images2 = load_data("data/day2_frames")
    output_dict_1 = inference(images1)
    output_dict_2 = inference(images2)
    estimate_global_poses(output_dict_1)
    estimate_global_poses(output_dict_2)
    save_output_dict(output_dict_1, "saved_reconstruction_day1.pkl")
    save_output_dict(output_dict_2, "saved_reconstruction_day2.pkl")
