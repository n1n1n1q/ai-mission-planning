#!/usr/bin/env python
# filepath: /Users/anya/adhd/recon3D/object_detection/yolo_test.py
"""
Test script for tracking people in a video using YOLO object detection.
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from recon3D.object_detection.detector import Detector


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Track people in a video using YOLO.')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to the input video file. If not provided, will use the default video.')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output video. If not provided, no output will be saved.')
    parser.add_argument('--show', action='store_true',
                        help='Show the video with detections in real-time.')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold for detections (default: 0.3).')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IOU threshold for NMS (default: 0.45).')
    parser.add_argument('--save-detections', type=str, default=None,
                        help='Path to save detection results in a text file.')
    parser.add_argument('--tracker', type=str, default="bytetrack.yaml", 
                        help='Tracker type to use (default: bytetrack.yaml).')
    parser.add_argument('--person-only', action='store_true',
                        help='Track only people (class 0) and ignore other objects.')
    
    return parser.parse_args()


def process_video_with_tracking(detector, video_path, output_path=None, show=False, 
                              tracker="bytetrack.yaml", person_only=False):
    """
    Process a video file with object tracking to track people.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_tracked_detections = []
    frame_idx = 0
    person_class = 0
    
    color_map = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{total_frames} ({(frame_idx/total_frames)*100:.1f}%)")
        
        results = detector.model.track(frame, persist=True, tracker=tracker)
        
        frame_detections = []
        
        for res in results:
            if res.boxes.id is not None:
                for i, box in enumerate(res.boxes):
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    detected_class = int(box.cls[0])
                    class_name = res.names[detected_class]
                    track_id = int(box.id[0])
                    
                    if person_only and detected_class != person_class:
                        continue
                        
                    frame_detections.append((track_id, x1, y1, x2, y2, conf, detected_class, class_name))
        
        all_tracked_detections.append(frame_detections)
        
        if output_path or show:
            frame_with_detections = frame.copy()
            
            for (track_id, x1, y1, x2, y2, conf, detected_class, class_name) in frame_detections:
                if track_id not in color_map:
                    hue = (track_id * 43) % 180
                    color_map[track_id] = tuple(int(x) for x in cv2.cvtColor(
                        np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0])
                
                color = color_map[track_id]
                
                cv2.rectangle(frame_with_detections, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                label = f"{class_name} #{track_id}: {conf:.2f}"
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                cv2.rectangle(frame_with_detections, 
                              (int(x1), int(y1)-text_height-baseline-5),
                              (int(x1)+text_width, int(y1)), 
                              color, -1)
                
                cv2.putText(frame_with_detections, label, (int(x1), int(y1)-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.putText(frame_with_detections, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if output_path:
                out.write(frame_with_detections)
            
            if show:
                cv2.imshow('Person Tracking', frame_with_detections)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        frame_idx += 1
    
    cap.release()
    if output_path:
        out.release()
    if show:
        cv2.destroyAllWindows()
    
    return all_tracked_detections


def main():
    """Main function to run person tracking on a video."""
    args = parse_arguments()
    
    if args.video is None:
        video_path = str(Path(__file__).resolve().parent.parent.parent / "IMG_9458.MOV")
        print(f"Using default video: {video_path}")
    else:
        video_path = args.video
    
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    print("Initializing YOLO detector...")
    classes = [0] if args.person_only else None
    detector = Detector(classes=classes)
    detector.set_parameters(conf=args.conf, iou=args.iou)
    
    print(f"Processing video: {video_path}")
    print("Press 'q' to quit during playback")
    
    start_time = time.time()
    
    all_detections = process_video_with_tracking(
        detector, 
        video_path, 
        output_path=args.output,
        show=args.show,
        tracker=args.tracker,
        person_only=args.person_only
    )
    
    end_time = time.time()
    
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    if args.save_detections and all_detections:
        save_path = args.save_detections
        print(f"Saving detection results to {save_path}")
        
        with open(save_path, 'w') as f:
            f.write("frame_id,track_id,x1,y1,x2,y2,confidence,class_id,class_name\n")
            for frame_id, frame_detections in enumerate(all_detections):
                for detection in frame_detections:
                    track_id, x1, y1, x2, y2, conf, class_id, class_name = detection
                    f.write(f"{frame_id},{track_id},{x1},{y1},{x2},{y2},{conf},{class_id},{class_name}\n")
    
    total_frames = len(all_detections)
    
    total_detections = sum(len(frame_dets) for frame_dets in all_detections)
    
    unique_tracks = set()
    for frame_dets in all_detections:
        for detection in frame_dets:
            track_id = detection[0]
            unique_tracks.add(track_id)
    
    avg_detections = total_detections / total_frames if total_frames > 0 else 0
    
    print(f"Total frames processed: {total_frames}")
    print(f"Total detections: {total_detections}")
    print(f"Unique tracks: {len(unique_tracks)}")
    print(f"Average detections per frame: {avg_detections:.2f}")


if __name__ == "__main__":
    main()