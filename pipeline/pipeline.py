import os

from pathlib import Path
from recon3D.reconstruction.video_splitter import video_to_frames


def video_to_point_cloud(video_path):
    frames_output_path = f"data/photos/frames_of_{Path(video_path).stem}"

    video_to_frames(video_path, frames_output_path, frames_per_second=1)


if __name__ == "__main__":
    video_file = "assets/hackaton videos /IMG_2265.MOV"
    video_to_point_cloud(video_file)
