import os
from image_model import *

PROGRAM_VIDEO_DIR = "static/videos/program"
os.makedirs(PROGRAM_VIDEO_DIR, exist_ok=True)


def similarity(path1, path2):
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

    # Compute cosine similarity
    return cosine_similarity_video(baseline, video2)
