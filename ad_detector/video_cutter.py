from dataclasses import dataclass

height = 270
width = 480
frame_size = height*width*3

@dataclass
class VideoCutter:
    input_video: str
    output_video: str
    start_frame: int
    frame_count: int

    def cut(self):
        i = open(self.input_video, 'rb')
        o = open(self.output_video, 'wb')

        i.seek(frame_size*self.start_frame)
        for _ in range(self.frame_count):
            b = i.read(frame_size)
            if b is None:
                break
            o.write(b)

        i.close()
        o.close()

iv = "dataset/Videos/data_test1.rgb"
ov = "dataset/Videos/starbucks_snapshot.rgb"
vc = VideoCutter(iv, ov, 5400, 1)
vc.cut()