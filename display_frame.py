import cv2
import numpy as np


# raw_bytes = np.fromfile('dataset/Videos/data_test1.rgb', np.dtype('B'))
raw_bytes = np.fromfile('test2/Videos/test2.rgb', np.dtype('B'))
n_frames = len(raw_bytes)//(270*480*3)
frames = raw_bytes.reshape((n_frames, 3, 270, 480)) # shape = (9000, 3, 270, 480)
frames = np.moveaxis(frames, 1, -1)  # pack rgb values per pixel to shape (9000, 270, 480, 3)

i = 4209
while True:
    print(i)
    frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
    cv2.imshow('', frame)
    key = cv2.waitKey(0)
    if key  & 0xFF == ord('m'):
        i += 1
    elif key & 0xFF == ord('n'):
        i -= 1
    elif key & 0xFF == ord('q'):
        break