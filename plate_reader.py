import argparse

import cv2
import re
import easyocr

import numpy as np
from tqdm import tqdm
from collections import Counter

from utils.image_utils import get_images_from_path, resize_image


class PlateReader:
    def __init__(self, detection_threshold=0.2, use_gpu=False):
        self._reader = easyocr.Reader (['en'], gpu=use_gpu)
        self._detection_threshold = detection_threshold

    @staticmethod
    def _format_license_text(detected_text: str) -> str:
        return re.sub(r'\D', '', detected_text)

    def read_license_plate(self, license_plate: np.ndarray or str) -> (str or None):
        if isinstance(license_plate, str):
            license_plate = cv2.imread(license_plate)
            if license_plate is None:
                raise ValueError(f"Could not read image from path: {license_plate}")

        license_plate = resize_image(license_plate)
        license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        license_plate = cv2.GaussianBlur(license_plate, (3, 3), 0)
        # perform ocr
        ocr_result = self._reader.readtext(license_plate)


        if len(ocr_result):
            (bbox, text, score) = ocr_result[0]

            formated_text = self._format_license_text(text)
            if (score > self._detection_threshold) and formated_text:
                result = formated_text
                return result

        return None

    @staticmethod
    def _post_process_multiple_plate_readings(plate_readings: list[str]) -> str:
        # Filter results to keep only strings of the same length as the most common length
        most_common_length = max(map(len, plate_readings), key=lambda x: sum(len(r) == x for r in plate_readings))
        filtered_results = [r for r in plate_readings if len(r) == most_common_length]

        # Majority voting for each character position
        final_result = ''.join(
            Counter(chars).most_common(1)[0][0]
            for chars in zip(*filtered_results)
        )
        return final_result


    def read_license_plate_multiple_images(self, images_path: str)-> str:
        plate_readings = []
        for image_path in tqdm(get_images_from_path(images_path)):
            ocr_result = self.read_license_plate(image_path)
            if ocr_result is not None:
                plate_readings.append(ocr_result)

        final_result = self._post_process_multiple_plate_readings(plate_readings)

        return final_result



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, help='path to the images of the plate')
    args = parser.parse_args()


    p = PlateReader()
    final_result = p.read_license_plate_multiple_images(args.images_path)
    print("Most likely result:", final_result)



