import os
from image_model import *
import subprocess
import numpy as np
from moviepy import VideoFileClip, CompositeVideoClip

PROGRAM_VIDEO_DIR = "static/videos/program"
os.makedirs(PROGRAM_VIDEO_DIR, exist_ok=True)

def convert_webm_to_mp4(input_path, output_path):
    """Convert a WebM file to MP4 format using ffmpeg"""
    try:
        command = [
            'ffmpeg', 
            '-i', input_path, 
            '-c:v', 'h264', 
            '-c:a', 'aac', 
            '-strict', 'experimental',
            output_path,
            '-y'  # Overwrite output file if it exists
        ]
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def similarity(path1, path2):
    # If path2 is a WebM file, convert it to MP4 first
    if path2.lower().endswith('.webm'):
        temp_mp4_path = path2.replace('.webm', '.mp4')
        if not convert_webm_to_mp4(path2, temp_mp4_path):
            # If conversion failed, try direct processing as a fallback
            print("WebM conversion failed, trying direct processing")
        else:
            path2 = temp_mp4_path
    
    # Process first video
    clip1 = VideoFileClip(path1)
    clip1 = clip1.subclipped(0, 3)
    final_video1 = CompositeVideoClip([clip1])
    jumpingjacks_path = os.path.join(PROGRAM_VIDEO_DIR, "jumpingjacks.mp4")
    
    # Write the first video
    final_video1.write_videofile(jumpingjacks_path)

    # Ensure the correct path is used
    frames_target, metadata_target = extract_frames(jumpingjacks_path, 4)
    detected_keypoints1, crop_region = extract_keypoints_and_crop(frames_target, metadata_target)
    baseline = np.array(detected_keypoints1)

    # Process second video
    clip2 = VideoFileClip(path2)
    clip2 = clip2.subclipped(0, 3)
    final_video2 = CompositeVideoClip([clip2])
    uservideo_path = os.path.join(PROGRAM_VIDEO_DIR, "uservideo.mp4")

    # Write the second video
    final_video2.write_videofile(uservideo_path)

    # Ensure the correct path is used
    frames_target, metadata_target = extract_frames(uservideo_path, 4)
    detected_keypoints2, crop_region = extract_keypoints_and_crop(frames_target, metadata_target)
    video2 = np.array(detected_keypoints2)

    # Cleanup temp MP4 file if created
    if path2.endswith('.mp4') and path2.replace('.mp4', '.webm') != path2:
        try:
            os.remove(path2)
        except:
            pass

    # Compute cosine similarity
    return dtw_similarity(baseline, video2)