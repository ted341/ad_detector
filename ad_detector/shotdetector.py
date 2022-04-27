from dataclasses import dataclass

# Standard PySceneDetect imports:
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import AdaptiveDetector


@dataclass
class Shot:
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    

class ShotDetector:
    def __init__(self, input_video, input_audio):
        self._input_path = './dataset/Videos/data_test1.mp4'
        self.threshold = 2.0
    
    def detect(self):
        # Create our video & scene managers, then add the detector.
        video_manager = VideoManager([self._input_path])
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
            shots.append(Shot(start_frame = timecode_start.get_frames(),
                              end_frame = timecode_end.get_frames(),
                              start_timestamp = timecode_start.get_seconds(),
                              end_timestamp = timecode_end.get_seconds()))
        breakpoint()
        return shots
    
    def debug(self):
        import json
        s = '[{"start_frame": 0, "end_frame": 606, "start_timestamp": 0.0, "end_timestamp": 20.2}, {"start_frame": 606, "end_frame": 1179, "start_timestamp": 20.2, "end_timestamp": 39.3}, {"start_frame": 1179, "end_frame": 2400, "start_timestamp": 39.3, "end_timestamp": 80.0}, {"start_frame": 2400, "end_frame": 2435, "start_timestamp": 80.0, "end_timestamp": 81.16666666666667}, {"start_frame": 2435, "end_frame": 2489, "start_timestamp": 81.16666666666667, "end_timestamp": 82.96666666666667}, {"start_frame": 2489, "end_frame": 2515, "start_timestamp": 82.96666666666667, "end_timestamp": 83.83333333333333}, {"start_frame": 2515, "end_frame": 2596, "start_timestamp": 83.83333333333333, "end_timestamp": 86.53333333333333}, {"start_frame": 2596, "end_frame": 2659, "start_timestamp": 86.53333333333333, "end_timestamp": 88.63333333333334}, {"start_frame": 2659, "end_frame": 2695, "start_timestamp": 88.63333333333334, "end_timestamp": 89.83333333333333}, {"start_frame": 2695, "end_frame": 2748, "start_timestamp": 89.83333333333333, "end_timestamp": 91.6}, {"start_frame": 2748, "end_frame": 2803, "start_timestamp": 91.6, "end_timestamp": 93.43333333333334}, {"start_frame": 2803, "end_frame": 2838, "start_timestamp": 93.43333333333334, "end_timestamp": 94.6}, {"start_frame": 2838, "end_frame": 3630, "start_timestamp": 94.6, "end_timestamp": 121.0}, {"start_frame": 3630, "end_frame": 4350, "start_timestamp": 121.0, "end_timestamp": 145.0}, {"start_frame": 4350, "end_frame": 5550, "start_timestamp": 145.0, "end_timestamp": 185.0}, {"start_frame": 5550, "end_frame": 5582, "start_timestamp": 185.0, "end_timestamp": 186.06666666666666}, {"start_frame": 5582, "end_frame": 5612, "start_timestamp": 186.06666666666666, "end_timestamp": 187.06666666666666}, {"start_frame": 5612, "end_frame": 5699, "start_timestamp": 187.06666666666666, "end_timestamp": 189.96666666666667}, {"start_frame": 5699, "end_frame": 5753, "start_timestamp": 189.96666666666667, "end_timestamp": 191.76666666666668}, {"start_frame": 5753, "end_frame": 5846, "start_timestamp": 191.76666666666668, "end_timestamp": 194.86666666666667}, {"start_frame": 5846, "end_frame": 5925, "start_timestamp": 194.86666666666667, "end_timestamp": 197.5}, {"start_frame": 5925, "end_frame": 5987, "start_timestamp": 197.5, "end_timestamp": 199.56666666666666}, {"start_frame": 5987, "end_frame": 6450, "start_timestamp": 199.56666666666666, "end_timestamp": 215.0}, {"start_frame": 6450, "end_frame": 7200, "start_timestamp": 215.0, "end_timestamp": 240.0}, {"start_frame": 7200, "end_frame": 9000, "start_timestamp": 240.0, "end_timestamp": 300.0}]'
        shots = []
        for shot_dict in json.loads(s):
            shots.append(Shot(**shot_dict))
        return shots
        