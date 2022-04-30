import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class VideoFormat(dict):
    height: int
    width: int


class LogoDetector:
    def __init__(self, video_path: str, logo_path: str, video_format: VideoFormat):
        self.video_path = video_path
        self.logo_path = logo_path
        self.video_format = video_format
        self.feature_detector = cv2.SIFT_create(
            nfeatures=1000,
            nOctaveLayers=3,
            contrastThreshold=0.09,
            edgeThreshold=16,
            sigma=1.6,
        )
        self.feature_matcher = cv2.BFMatcher()
        self.gamma_value = 1.75
        self.gamma_lookup = np.array(
            [
                np.clip(pow(i / 255.0, self.gamma_value) * 255.0, 0, 255)
                for i in range(256)
            ],
            dtype=np.uint8,
        ).reshape((1, 256))

    def __apply_gamma_correction(self, image):
        return cv2.LUT(image, self.gamma_lookup)

    def __detect_features(self, image):
        return self.feature_detector.detectAndCompute(image, None)

    def __match_descriptors(self, query_desc, train_desc):
        return [
            [m]
            for m, n in self.feature_matcher.knnMatch(query_desc, train_desc, k=2)
            if m.distance < 0.7 * n.distance
        ]

    def __draw_bounding_box(self, frame, matches, query_kps, train_kps):
        if len(matches) < 4:
            return
        """
        # Collect points of matching features from query and train images
        src_pts = np.empty((1, len(matches)), dtype=np.float32)
        dst_pts = np.empty((1, len(matches)), dtype=np.float32)
        for m in matches:
            np.append(src_pts, query_kps[m[0].queryIdx].pt)
            np.append(dst_pts, train_kps[m[0].trainIdx].pt)
        print(src_pts)
        src_pts = np.reshape(src_pts, (-1, 1, 2))
        dst_pts = np.reshape(dst_pts, (-1, 1, 2))
        """
        src_pts = np.float32([query_kps[m[0].queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([train_kps[m[0].trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        # Calculate transformation matrix
        matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if matrix is not None:
            # apply perspective transform to train image corners to get a bounding box coordinates on a sample image
            shape = (self.video_format.height, self.video_format.width)
            # map four bounding points of query image to train image based on the perspective (might not be rectangle on train image)
            map_pts = cv2.perspectiveTransform(
                np.float32(
                    [
                        (0, 0),
                        (0, shape[0] - 1),
                        (shape[1] - 1, shape[0] - 1),
                        (shape[1] - 1, 0),
                    ]
                ).reshape(-1, 1, 2),
                matrix,
            )
            # find minimum rectangle of the mapping points on train image
            rect = cv2.minAreaRect(map_pts)
            # generate four bounding points
            box = cv2.boxPoints(rect)
            # convert to integers
            box = np.int0(box)
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

    def run(self):

        video_file = open(self.video_path, "rb")
        frame_size = self.video_format.height * self.video_format.width * 3

        # Read logo image into gray scale for feature matching
        logo_image = cv2.imread(self.logo_path, cv2.IMREAD_GRAYSCALE)
        # Generate features from the image
        logo_kps, logo_desc = self.__detect_features(logo_image)

        # Read one frame at a time
        while raw := video_file.read(frame_size):

            # Convert byte array to ndarray which is compatible with OpenCV Mat data type
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (3, self.video_format.height, self.video_format.width)
            )
            # Pack RGB values by pixels
            frame = np.moveaxis(frame, 0, -1)
            # Convert RGB to BGR (cannot flip by [::-1])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Store filtered frame elsewhere so the original frame isn't modified
            tmp_frame = self.__apply_gamma_correction(frame)
            # Convert image to gray scale
            tmp_frame = cv2.cvtColor(tmp_frame, cv2.COLOR_BGR2GRAY)
            # Generate features from the image
            frame_kps, frame_desc = self.__detect_features(tmp_frame)
            # Conduct Knn matching
            matches = self.__match_descriptors(logo_desc, frame_desc)
            # Highlight the logo on the frame
            self.__draw_bounding_box(frame, matches, logo_kps, frame_kps)
            # Display the resulting frame
            cv2.imshow("Video", frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            # Output ???

        # Close all the frames
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        video_file.close()
