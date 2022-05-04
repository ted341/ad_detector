import numpy as np
import time
import wave
import pyaudio
import cv2


class VideoPlayer:
    def __init__(self, input_video, input_audio, config):
        self._input_video = open(input_video, "rb")
        self._input_audio = wave.open(input_audio, "rb")
        self._player = pyaudio.PyAudio()
        self._output_stream = self._player.open(
            format=self._player.get_format_from_width(self._input_audio.getsampwidth()),
            channels=self._input_audio.getnchannels(),
            rate=self._input_audio.getframerate(),
            output=True,
        )

        self._video_height = config["video"]["height"]
        self._video_width = config["video"]["width"]
        self._video_frame_rate = config["video"]["frame_rate"] or 30

    def play(self):

        interval = 1 / self._video_frame_rate
        frame_size = self._video_height * self._video_width * 3
        samples_per_frame = (
            self._input_audio.getframerate() // self._video_frame_rate
        )

        while True:
            start_time = time.time()
            # Read one frame at a time
            if (raw_frame := self._input_video.read(frame_size)) is None:
                break
            # Convert byte array to ndarray which is compatible with OpenCV Mat data type
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                (3, self._video_height, self._video_width)
            )
            # Pack RGB values by pixels
            frame = np.moveaxis(frame[::-1], 0, -1)
            cv2.imshow("demo", frame)
            key = cv2.waitKey(10)
            # Handle pressed keys
            if key & 0xFF == ord("q"):  # stop
                break
            elif key & 0xFF == ord(" "):  # pause
                while cv2.waitKey(0) & 0xFF != ord(" "):
                    pass
            # write audio stream
            self._output_stream.write(
                self._input_audio.readframes(samples_per_frame)
            )
            # sleep for a while if needed
            if (wait_time := interval - (time.time() - start_time) - 10) > 0:
                time.sleep(wait_time)

        cv2.destroyAllWindows()
        cv2.waitKey(1)

        self._output_stream.stop_stream()
        self._output_stream.close()
        self._player.terminate()
        self._input_video.close()
        self._input_audio.close()
