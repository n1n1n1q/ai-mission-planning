import cv2
import os


def video_to_frames(video_path, output_folder, frames_per_second=1):
    try:
        os.makedirs(output_folder, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {output_folder}: {e}")
        return

    video_capture = cv2.VideoCapture(video_path)

    total_frames = 0
    saved_frame_count = 0
    num_of_skipped_frames = int(video_capture.get(cv2.CAP_PROP_FPS) / frames_per_second)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        if total_frames % num_of_skipped_frames == 0:
            frame_path = os.path.join(
                output_folder, f"frame_{saved_frame_count:04d}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1

        total_frames += 1

    video_capture.release()


if __name__ == "__main__":
    video_file = "assets/hackaton videos /IMG_2265.MOV"
    output_dir = "recon3D/data/photos/"
    video_to_frames(video_file, output_dir)
