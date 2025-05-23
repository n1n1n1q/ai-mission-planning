"""
Test SLAM
"""

import os
import cv2
from planner.slam.visual_slam import VisualSLAM
from planner.slam.utils import DetectedObject

OUTPUT_DIR = "./output_images"
IN_FILE = "video.MOV"


os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_keypoints_and_bboxes(frame, keypoints, objects, prev_keypoints=[], prev_matches=[]):
    """
    Draw keypoints and bounding boxes on the frame
    
    Args:
        frame: Input frame
        keypoints: List of keypoints
        objects: List of DetectedObject instances
        
    Returns:
        Frame with visualizations
    """
    vis_frame = frame.copy()

    for obj in objects:
        x, y, w, h = obj.bbox
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            vis_frame, 
            f"{obj.class_name}: {obj.confidence:.2f}", 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        color = (255, 0, 0)
        for obj in objects:
            if x >= obj.bbox[0] and x <= obj.bbox[0] + obj.bbox[2] and \
               y >= obj.bbox[1] and y <= obj.bbox[1] + obj.bbox[3]:
                color = (0, 0, 255)
                break
        
        cv2.circle(vis_frame, (x, y), 3, color, -1)
    
    if prev_keypoints and prev_matches:
        for i, kp in enumerate(prev_keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if prev_matches[i] is not None:
                x_prev, y_prev = int(prev_matches[i].pt[0]), int(prev_matches[i].pt[1])
                cv2.line(vis_frame, (x, y), (x_prev, y_prev), (255, 255, 0), 1)

    return vis_frame

def process_image_file(slam, img_path):
    """
    Process a single image file
    
    Args:
        slam: VisualSLAM instance
        img_path: Path to the image file
    """
    frame = cv2.imread(img_path)
    
    if frame is None:
        print(f"Could not read image: {img_path}")
        return
        
    vis_frame, detected_objects = slam.process_frame(frame)
    keypoints = slam.prev_kps if slam.prev_kps is not None else []

    custom_vis = draw_keypoints_and_bboxes(frame, keypoints, detected_objects)
    
    cv2.imshow("SLAM Visualization", custom_vis)
    
    img_name = os.path.basename(img_path)
    output_path = os.path.join(OUTPUT_DIR, f"processed_{img_name}")
    cv2.imwrite(output_path, custom_vis)
    print(f"Saved: {output_path}")
    
    return cv2.waitKey(1) & 0xFF

def process_video_file(slam, video_path):
    """
    Process a video file
    
    Args:
        slam: VisualSLAM instance
        video_path: Path to the video file
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_name = os.path.basename(video_path)
    output_path = os.path.join(OUTPUT_DIR, f"processed_{video_name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    prev_keypoints, prev_matches = [], []
    prev_des = None
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vis_frame, detected_objects = slam.process_frame(frame)
        keypoints = slam.prev_kps if slam.prev_kps is not None else []

        curr_matches = []
        if prev_keypoints and keypoints and len(prev_keypoints) > 0 and len(keypoints) > 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create()
            _, curr_des = orb.compute(gray_frame, keypoints)
            
            if frame_count > 0 and prev_des is not None and curr_des is not None:
                from planner.slam.utils import keypoints_matcher
                matches = keypoints_matcher(prev_des, curr_des)
                curr_matches = [None] * len(prev_keypoints)
                for match in matches:
                    curr_matches[match.queryIdx] = keypoints[match.trainIdx]
        
        custom_vis = draw_keypoints_and_bboxes(frame, keypoints, detected_objects, prev_keypoints, curr_matches)
    
        prev_keypoints = keypoints
        prev_matches = curr_matches
    
        if keypoints:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if 'gray_frame' not in locals() else gray_frame
            orb = cv2.ORB_create() if 'orb' not in locals() else orb
            _, prev_des = orb.compute(gray_frame, keypoints)
        else:
            prev_des = None
            
        out.write(custom_vis)
        cv2.imshow("SLAM Visualization", custom_vis)
        
        if frame_count % 30 == 0:
            img_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(img_path, custom_vis)
            print(f"Saved frame: {img_path}")
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    print(f"Processed video saved to: {output_path}")

if __name__ == "__main__":
    slam = VisualSLAM()
    cv2.namedWindow("SLAM Visualization", cv2.WINDOW_NORMAL)
    
    mov_file = IN_FILE
    if os.path.exists(mov_file):
        print(f"Processing video file: {mov_file}")
        process_video_file(slam, mov_file)
    else:
        print("Video file not found. Processing images instead...")
        for img_name in os.listdir("images"):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join("images", img_name)
            if process_image_file(slam, img_path) == ord('q'):
                break
    
    cv2.destroyAllWindows()