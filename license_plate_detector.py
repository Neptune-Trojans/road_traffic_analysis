import argparse
from dataclasses import dataclass

import os.path
import cv2
from tqdm import tqdm
from ultralytics import YOLO

from utils.folder_utils import get_app_path, get_subfolders
from utils.image_utils import get_images_from_path, crop_image

# https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8

@dataclass
class LicensePlateData:
    x1: int
    y1: int
    x2: int
    y2: int
    original_image_path: str
    plate_crop_image_path: str

class LicencePlateDetector:
    def __init__(self, weights_path: str):
        self._detector =  YOLO(weights_path)
        # self._base_path = base_path

    def _crop_image_path(self, output_base_path: str, original_file_name: str, detection_id: int) -> str:
        parts = original_file_name.split('/')
        track_id = parts[-2]
        file_name = parts[-1]
        file_name_no_extension = file_name.split('.')[0]

        crop_image_path = os.path.join(output_base_path, f"{track_id}_{file_name_no_extension}_plate_{detection_id}.jpg")
        return crop_image_path

    def detect(self, image_path: str, output_path: str=None) -> [LicensePlateData]:
        results = []
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded.")

        os.makedirs(output_path, exist_ok=True)

        license_plates = self._detector(image, verbose=False)[0].boxes.data.tolist()

        for i, (x1, y1, x2, y2, score, class_id) in enumerate(license_plates):
            license_plate_image = crop_image(image, [x1, y1, x2, y2])
            crop_image_path = self._crop_image_path(output_path, image_path, i)

            cv2.imwrite(crop_image_path, license_plate_image)
            results.append(LicensePlateData(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                                            original_image_path=image_path, plate_crop_image_path=crop_image_path))

        return results

if __name__ == '__main__':
    app_path = get_app_path('plate_detection')
    weights_path = 'models/license_plate_detector.pt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, help='input video path')
    args = parser.parse_args()

    detector = LicencePlateDetector(weights_path)

    folders = get_subfolders(args.images_path)
    for folder in tqdm(folders):
        folder_name = os.path.basename(folder)
        output_path = os.path.join(app_path, folder_name)
        for image_path in get_images_from_path(folder):
            detector.detect(image_path, output_path)


