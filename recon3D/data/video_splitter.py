import cv2
import os


def video_to_frames(video_path, output_folder, frames_per_second=1):
    try:
        os.makedirs(output_folder, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {output_folder}: {e}")
        return

    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    total_frames = 0
    saved_frame_count = 0
    num_of_skipped_frames = int(fps / frames_per_second)

    frame_time_intervals = []

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        if total_frames % num_of_skipped_frames == 0:
            frame_path = os.path.join(
                output_folder, f"frame_{saved_frame_count:04d}.jpg"
            )
            cv2.imwrite(frame_path, frame)

            current_time = total_frames / fps
            start_time = current_time
            end_time = (total_frames + num_of_skipped_frames) / fps

            frame_time_intervals.append((start_time, end_time))

            saved_frame_count += 1

        total_frames += 1

    video_capture.release()

    return frame_time_intervals, saved_frame_count


if __name__ == "__main__":
    video_file = "assets/hackaton videos /IMG_2265.MOV"
    output_dir = "recon3D/data/photos/"
    time_intervals, num_frames = video_to_frames(video_file, output_dir)

    print(time_intervals)
    print(num_frames)
