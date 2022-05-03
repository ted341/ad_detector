import cv2
import numpy as np


class LogoDetector:
    def __init__(self, input_video: str, output_video: str, config: dict):
        self.input_video = open(input_video, "rb")
        self.output_video = open(output_video, "wb")

        self.input_logos = config["logo"]["paths"] or []
        self.video_height = config["video"]["height"] or 270
        self.video_width = config["video"]["width"] or 480

        self.feature_detector = cv2.SIFT_create(**config["detect"]["sift"])
        self.feature_matcher = cv2.BFMatcher()

        self.ratio = config["detect"]["ratio"] or 0.7
        self.saturation = config["detect"]["saturation"] or 1.15
        self.gamma_value = config["detect"]["gamma"] or 1.6
        self.gamma_lookup = np.array(
            [
                np.clip(pow(i / 255.0, self.gamma_value) * 255.0, 0, 255)
                for i in range(256)
            ],
            dtype=np.uint8,
        ).reshape(1, 256)

        self.show_result = config["detect"]["debug"] or False
        self.write_output_file = config["detect"]["export"] or False
        self.testcase = config["demo"]["testcase"] or 0

    def __apply_gamma_correction(self, bgr):
        return cv2.LUT(bgr, self.gamma_lookup)

    def __increase_saturation(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * self.saturation
        # hsv[:,:,2] = hsv[:,:,2]*1 # brightness
        cv2.inRange(hsv, (0, 0, 0), (255, 255, 255))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def __detect_features(self, image):
        return self.feature_detector.detectAndCompute(image, None)

    def __match_descriptors(self, query_desc, train_desc):
        matches = []
        for m in self.feature_matcher.knnMatch(query_desc, train_desc, k=2):
            if len(m) == 2 and m[0].distance < self.ratio * m[1].distance:
                matches.append([m[0]])
        return matches

    def __draw_bounding_box(self, frame, matches, query_kps, train_kps):
        # Skip if too few match points
        if len(matches) < 8:
            return False

        # Collect points of matching features from query and train images
        src_pts = np.float32([query_kps[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([train_kps[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Calculate transformation matrix
        if (matrix := cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]) is not None:
            # map four corners of query image to train image based on the perspective (might not look rectangular)
            shape = (self.video_height, self.video_width)
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
            c, (w, h), a = cv2.minAreaRect(map_pts)
            # remove false positives
            if (
                w == 0
                or h == 0
                or w / h <= 1 / 3
                or w / h >= 3
                or w < 30
                or h < 30
                or (a >= 10 and a <= 80)
            ):
                return False
            # generate four bounding points
            box = np.int0(cv2.boxPoints((c, (w, h), a)))
            # remove false positive
            if not (
                np.all(box[:, 0] >= 0)
                or np.all(box[:, 0] < shape[1])
                or np.all(box[:, 1] >= 0)
                or np.all(box[:, 1] < shape[0])
            ):
                return False
            # draw bounding box
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

        return True

    def __export(self, frame):
        self.output_video.write(frame.tobytes())

    def run(self):

        result = []
        frame_size = self.video_height * self.video_width * 3

        # logo 1
        logo_image = cv2.imread(self.input_logos[self.testcase * 2], cv2.IMREAD_COLOR)
        logo_kps, logo_desc = self.__detect_features(logo_image)

        # logo 2
        logo2_image = cv2.imread(self.input_logos[self.testcase * 2 + 1], cv2.IMREAD_COLOR)
        logo2_kps, logo2_desc = self.__detect_features(logo2_image)

        # Do processing frame by frame
        while raw := self.input_video.read(frame_size):
            # Convert byte array to ndarray which is compatible with OpenCV Mat data type
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (3, self.video_height, self.video_width)
            )
            # Pack RGB values by pixels
            frame = np.moveaxis(frame, 0, -1)
            # Convert RGB to BGR (cannot flip by [::-1])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Adjust color of the frame
            # tmp_frame = self.__apply_gamma_correction(frame)
            tmp_frame = self.__increase_saturation(frame)
            # Generate features from the image
            frame_kps, frame_desc = self.__detect_features(tmp_frame)
            # Perform Knn matching
            matches = self.__match_descriptors(logo_desc, frame_desc)
            # Highlight the logo on the frame
            matched = self.__draw_bounding_box(frame, matches, logo_kps, frame_kps)

            if not matched:
                # Perform Knn matching
                matches = self.__match_descriptors(logo2_desc, frame_desc)
                # Highlight the logo on the frame
                self.__draw_bounding_box(frame, matches, logo2_kps, frame_kps)

            if self.show_result:
                # Display the resulting frame
                cv2.imshow("Video", frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.moveaxis(frame, -1, 0)

            if self.write_output_file:
                self.__export(frame)

            result.append(frame)

        # Close all the frames
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # Close files
        self.input_video.close()
        self.output_video.close()

        return result
