import os
from dataclasses import asdict
import json

from tqdm import tqdm
import yaml
import numpy as np
import ffmpeg
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import AdaptiveDetector

from ad_detector.shot import Shot


class ShotDetector:
    def __init__(self, video_name, frames):
        self._mp4_path = video_name[:-3] + 'mp4'
        self.threshold = 3.0
        self._preprocess_video(frames)

    def _preprocess_video(self, frames):
        """ Convert .rgb and .wav file to .mp4 format for scenedetect API"""
        if os.path.exists(self._mp4_path):
            print('\tmp4 file already exists')
            return
        
        with open("config.yaml") as file:
            config = yaml.safe_load(file)
            height = config['video']['height']
            width = config['video']['width']
            fps = config['video']['frame_rate']
        
        print('\tstart converting to mp4...')
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', framerate=fps)
            .output(self._mp4_path, pix_fmt='yuv420p', vcodec="libx264")
            .overwrite_output()
            .run_async(pipe_stdin=True, overwrite_output=True, pipe_stderr=True)
        )

        for frame in tqdm(frames):
            process.stdin.write(frame.astype(np.uint8).tobytes())
            
        process.stdin.close()
        process.wait()

    def detect(self, save_json=False):
        # Create our video & scene managers, then add the detector.
        video_manager = VideoManager([self._mp4_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector(video_manager, 
                                                    adaptive_threshold=self.threshold))

        # Improve processing speed by downscaling before processing.
        video_manager.set_downscale_factor()

        # Start the video manager and perform the scene detection.
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        # Each returned scene is a tuple of the (start, end) timecode.
        shots = []
        for (timecode_start, timecode_end) in scene_manager.get_scene_list():
            shots.append(Shot(sequence = len(shots),
                              start_frame = timecode_start.get_frames(),
                              end_frame = timecode_end.get_frames(),
                              start_timestamp = timecode_start.get_seconds(),
                              end_timestamp = timecode_end.get_seconds()))
        
        if save_json:
            json_list = [asdict(shot) for shot in shots]
            with open('./shot-result/new-dataset.json', 'w') as file:
                json.dump(json_list, file)
            
        
        return shots
    
    def detect_from_json(self, i):
        with open(f'./shot-result/test2.json') as file:
            shot_raw = json.load(file)
        return [ Shot(**shot_dict) for shot_dict in shot_raw ]