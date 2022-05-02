import cv2
import numpy as np


class LogoDetector:
    def __init__(self, input_video: str, output_video: str, config: dict):
        self.input_video = open(input_video, "rb")
        self.output_video = open(output_video, "wb")

        self.input_logos = config["logo"]["paths"]
        self.video_height = config["video"]["height"] or 270
        self.video_width = config["video"]["width"] or 480

        self.feature_detector = cv2.SIFT_create(**config["detect"]["sift"])
        self.feature_matcher = cv2.BFMatcher()

        self.gamma_value = config["detect"]["gamma"]
        self.gamma_lookup = np.array(
            [
                np.clip(pow(i / 255.0, self.gamma_value) * 255.0, 0, 255)
                for i in range(256)
            ],
            dtype=np.uint8,
        ).reshape(1, 256)

        self.show_result = config["detect"]["debug"]
        self.write_output_file = config["detect"]["export"]

    def __apply_gamma_correction(self, image):
        return cv2.LUT(image, self.gamma_lookup)

    def __detect_features(self, image):
        return self.feature_detector.detectAndCompute(image, None)

    def __match_descriptors(self, query_desc, train_desc):
        matches = [
            [first]
            for first, second in self.feature_matcher.knnMatch(
                query_desc, train_desc, k=2
            )
            if first.distance < 0.7 * second.distance
        ]
        return matches

    def __draw_bounding_box(self, frame, matches, query_kps, train_kps):
        if len(matches) < 4:
            return

        # Collect points of matching features from query and train images
        src_pts = np.float32([query_kps[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([train_kps[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Calculate transformation matrix
        matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        #print(matrix)
        if matrix is not None:
            # apply perspective transform to train image corners to get a bounding box coordinates on a sample image
            shape = (self.video_height, self.video_width)
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

    def __export(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.moveaxis(frame, -1, 0)
        self.output_video.write(frame.tobytes())

    def run(self):

        result = []
        frame_size = self.video_height * self.video_width * 3
        # Read logo image into gray scale for feature matching
        logo_image = cv2.imread(self.input_logos[5], cv2.IMREAD_GRAYSCALE)
        # Generate features from the image
        logo_kps, logo_desc = self.__detect_features(logo_image)

        # Read one frame at a time
        while raw := self.input_video.read(frame_size):
            # Convert byte array to ndarray which is compatible with OpenCV Mat data type
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (3, self.video_height, self.video_width)
            )
            # Pack RGB values by pixels
            frame = np.moveaxis(frame, 0, -1)
            # Convert RGB to BGR (cannot flip by [::-1])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #print(frame)
            # Store filtered frame elsewhere so the original frame isn't modified
            tmp_frame = self.__apply_gamma_correction(frame)
            #print(tmp_frame)
            # Convert image to gray scale
            tmp_frame = cv2.cvtColor(tmp_frame, cv2.COLOR_BGR2GRAY)
            #print(tmp_frame)
            # Generate features from the image
            frame_kps, frame_desc = self.__detect_features(tmp_frame)
            # Conduct Knn matching
            matches = self.__match_descriptors(logo_desc, frame_desc)
            # Highlight the logo on the frame
            self.__draw_bounding_box(frame, matches, logo_kps, frame_kps)
            
            if self.show_result:
                # Display the resulting frame
                cv2.imshow("Video", frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
            
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
