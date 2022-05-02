import os
from dataclasses import dataclass, field, asdict
import json

from tqdm import tqdm
import yaml
import numpy as np
import ffmpeg
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import AdaptiveDetector


@dataclass
class Shot:
    sequence: int
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    features: dict = field(default_factory=lambda : {})
    test_is_ad: bool = False
    is_ad: bool = None
    
    @property
    def duration(self):
        return self.end_timestamp - self.start_timestamp
    
    

class ShotDetector:
    def __init__(self, input_video):
        self.input_video = input_video
        self._mp4_path = input_video[:-3] + 'mp4'
        self.threshold = 3.0
        self._preprocess_video()

    def _preprocess_video(self):
        """ Convert .rgb and .wav file to .mp4 format for scenedetect API"""
        if os.path.exists(self._mp4_path):
            print('mp4 file already exists')
            return
        
        with open("format.yaml") as file:
            format = yaml.safe_load(file)
            height = format['video_height']
            width = format['video_width']
            fps = format['video_fps']
        
        print('Start converting...')
        raw_bytes = np.fromfile(self.input_video, np.dtype('B'))
        n_frames = len(raw_bytes)//(height*width*3)
        frames = raw_bytes.reshape((n_frames, 3, height, width))
        frames = np.moveaxis(frames, 1, -1)  # pack rgb values per pixel
        
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
            print('Saving to json')
            json_list = [asdict(shot) for shot in shots]
            with open('./shot-result/new-dataset.json', 'w') as file:
                json.dump(json_list, file)
            
        
        return shots
    
    def from_json(self, i):
        with open(f'./shot-result/dataset{i}.json') as file:
            shot_raw = json.load(file)
        return [ Shot(**shot_dict) for shot_dict in shot_raw ]