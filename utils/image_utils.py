import os
import cv2
import numpy as np


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

def crop_image(image: np.ndarray, bbox: list) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]


def draw_bounding_box(image: np.ndarray, bbox: list) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image

def get_images_from_path(path: str) -> list[str]:
    images = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(IMAGE_EXTENSIONS):
            images.append(file_path)
    return images

def resize_image(image: np.ndarray, target_width = 300) -> np.ndarray:

    height, width = image.shape[:2]
    scaling_factor = target_width / width

    target_height = int(height * scaling_factor)
    return cv2.resize(image, (target_width, target_height))
