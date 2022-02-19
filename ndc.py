import os
import numpy as np
import random
import cv2 as cv


class NDC:
    """
    [Work in Progress]
    Custom written Data Augmentation Algorithm called NDC (New Data by Combining)
    """

    def __init__(self, df: list, data_folder: str, img_height: int = 256, img_width: int = 256):
        self.data_img_list = df
        self.data_folder = data_folder
        self.processed_data_list = []
        self.processed_imgs = []
        self.img_height = img_height
        self.img_width = img_width

    def run(self):
        print(f"[NDC] Augmenting Dataset : {self.data_folder}")
        self.process()

    def process(
        self, subclass: str = "", save_horizontal_flip=True, save_vertical_flip=True, save_both_axises_flip=True
    ) -> list:
        """
        Combines 2 images into one with optional transforming
        """
        idx = 0
        for img in tqdm(self.data_img_list):
            # Step 1 - Get two samples
            img1 = img[0]

            # Get min & max index from image list
            max_idx = len(self.data_img_list) - 1

            # Select random image as second image to combine with
            unused_imgs = [i for i in range(0, max_idx) if i not in self.processed_imgs and i != idx]
            if len(unused_imgs) == 0:
                continue
            img2_idx = random.choice(unused_imgs)
            img2 = self.data_img_list[img2_idx][0]

            # Step 2 - Adjust alpha channel of both images to 0.5
            alpha = np.full((self.img_height, self.img_width), 128, dtype=np.uint8)
            img1 = np.dstack((img1, alpha))
            img2 = np.dstack((img2, alpha))

            # Step 3 - Combine the 2 images
            combined_img = np.concatenate((img1, img2), axis=0)

            # Step 4 - Duplicate the image values 4 times
            combined_img = np.multiply(combined_img, 4.0)

            # Step 5 - Save & write result to the data folder
            result_img = combined_img
            self.save_img(result_img, idx)

            # Flip Image (optional)
            if save_horizontal_flip:
                self.save_img(self.flip_image(result_img, 1), idx)
            if save_vertical_flip:
                self.save_img(self.flip_image(result_img, 0), idx)
            if save_both_axises_flip:
                self.save_img(self.flip_image(result_img, -1), idx)

            idx += 1

        return self.processed_data_list

    def flip_image(self, img: np.array, axis: int = 0) -> np.array:
        """
        Axis 1 is around the X-axis
        Axis 0 is around the Y-axis
        Axis -1 is around both axises
        """
        flipped_img = cv.flip(img, axis)

        return flipped_img

    def save_img(self, img: np.array, idx: int):
        """
        Save image into list
        """
        self.processed_data_list.append(img)
        self.processed_imgs.append(idx)

        # write image
        img_name = f"ndc_{idx}.jpg"
        saved_img_path = os.path.join(self.data_folder, img_name)
        cv.imwrite(saved_img_path, img)
