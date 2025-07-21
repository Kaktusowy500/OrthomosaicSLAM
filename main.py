import cv2
import numpy as np
from imutils import paths
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class Orthomosaic:
    def __init__(self):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        self.no_raw_images = []
        self.temp_image = []
        self.final_image = []

        self.ap = argparse.ArgumentParser()
        self.ap.add_argument(
            "-i",
            "--images",
            type=str,
            help="path to input directory of images to stitch",
        )
        self.ap.add_argument(
            "-o", "--output", type=str, required=True, help="path to the output image"
        )
        self.ap.add_argument("-v", "--video", type=str, help="path to video input")
        self.ap.add_argument("--debug", action="store_true", help="enable debug mode")
        self.args = vars(self.ap.parse_args())
        self.debug = self.args.get("debug", False)

    def run(self):
        if self.args.get("images"):
            self.load_dataset()
            self.mixer()
        elif self.args.get("video"):
            self.from_video()
        else:
            logging.error("No input provided. Please specify images or video.")
            return

    def scale_image(self, image, scale):
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dim = (width, height)
        return cv2.resize(image, dim)

    def load_dataset(self):
        # grab the paths to the input images and initialize our images list
        if self.debug:
            logging.info("Importing Images...")
        self.imagePaths = sorted(list(paths.list_images(self.args["images"])))
        self.images = []
        for imagePath in self.imagePaths:
            image = cv2.imread(imagePath)
            processed_image = self.scale_image(image, 0.5)
            self.images.append(processed_image)
        if self.debug:
            logging.info("Importing Complete")

    def from_video(self):
        if self.debug:
            logging.info("Importing Video...")
        self.cap = cv2.VideoCapture(self.args["video"])
        self.nth_frame = 20
        self.images = []
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning(
                    "Failed to read frame from video or end of video reached"
                )
                break
            frame_count += 1
            frame = self.scale_image(frame, 0.25)
            cv2.imshow("Video Frame", frame)
            if frame_count % self.nth_frame == 0:
                self.images.append(frame)
                if len(self.images) > 1:
                    if len(self.images) == 2:
                        self.temp_image = self.sticher(self.images[-2], self.images[-1])
                    else:
                        self.temp_image = self.sticher(self.temp_image, self.images[-1])
                    cv2.imshow("output_temp", self.temp_image)
                    cv2.waitKey(500)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        self.final_image = self.temp_image
        cv2.imshow("output", self.final_image)
        cv2.imwrite(self.args["output"], self.final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.cap.release()

    def mixer(self):
        self.no_raw_images = len(self.images)
        if self.debug:
            logging.info(f"{self.no_raw_images} Images have been loaded")
        for x in range(self.no_raw_images):
            if x == 0:
                self.temp_image = self.sticher(self.images[x], self.images[x + 1])
            elif x < self.no_raw_images - 1:
                self.temp_image = self.sticher(self.temp_image, self.images[x + 1])
            else:
                self.final_image = self.temp_image
            cv2.imshow("output_temp", self.temp_image)
            cv2.waitKey(200)

        cv2.imshow("output", self.final_image)
        cv2.imwrite(self.args["output"], self.final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sticher(self, image1, image2):
        orb = cv2.ORB_create(nfeatures=1000)
        logging.info(f"Processing image with shape: {image1.shape}")

        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        all_matches = []
        for m, n in matches:
            all_matches.append(m)

        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)

        # Set minimum match condition
        MIN_MATCH_COUNT = 0

        if len(good) > MIN_MATCH_COUNT:
            # Convert keypoints to an argument for findHomography
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(
                -1, 1, 2
            )

            # Establish a homography
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            result = self.wrap_images(image2, image1, M)
            return result
        else:
            logging.error("Not enough matches found")

    def wrap_images(self, image1, image2, H):
        rows1, cols1 = image1.shape[:2]
        rows2, cols2 = image2.shape[:2]
        list_of_points_1 = np.float32(
            [[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]
        ).reshape(-1, 1, 2)
        temp_points = np.float32(
            [[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]
        ).reshape(-1, 1, 2)
        list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
        list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)
        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

        translation_dist = [-x_min, -y_min]

        H_translation = np.array(
            [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]
        )
        output_img = cv2.warpPerspective(
            image2, H_translation.dot(H), (x_max - x_min, y_max - y_min)
        )
        output_img[
            translation_dist[1] : rows1 + translation_dist[1],
            translation_dist[0] : cols1 + translation_dist[0],
        ] = image1
        return output_img


if __name__ == "__main__":
    tester = Orthomosaic()
    tester.run()
