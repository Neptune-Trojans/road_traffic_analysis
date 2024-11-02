import os.path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from utils.folder_utils import get_app_path
from utils.image_utils import get_images_from_path, crop_image

# https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8

class LicencePlateDetector:
    def __init__(self, weights_path, base_path):
        self._detector =  YOLO(weights_path)
        self._base_path = base_path

    def detect(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded.")

        license_plates = self._detector(image, verbose=False)[0].boxes.data.tolist()

        if not license_plates:
            print("No license plates detected.")

        for i, (x1, y1, x2, y2, score, class_id) in enumerate(license_plates):
            license_plate_image = crop_image(image, [x1, y1, x2, y2])
            output_path = os.path.join(self._base_path, f"{os.path.basename(image_path).split('.')[0]}_plate_{i}.jpg")
            cv2.imwrite(output_path, license_plate_image)

if __name__ == '__main__':
    app_path = get_app_path('plate_detection')
    weights_path = 'models/license_plate_detector.pt'

    detector = LicencePlateDetector(weights_path, app_path)

    images_path =  '/Users/yudkin/.flamingo/Applications/video_tracker/65'

    for image_path in tqdm(get_images_from_path(images_path)):
        detector.detect(image_path)


