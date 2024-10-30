import argparse

import cv2
import os

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from utils.folder_utils import get_app_path
from utils.image_utils import crop_image

class VehicleDetectionPipeline:
    def __init__(self, video_path: str, base_path: str):
        self._det_model = YOLO("yolo11n.pt")
        self._video_path = video_path
        self._base_path = base_path

    @property
    def annotated_video_path(self):
        file_name = os.path.basename(self._video_path)
        return os.path.join(app_path, file_name)

    def _perform_detection(self, frame: np.ndarray):
        results = self._det_model.track(frame, persist=True, tracker='botsort.yaml', iou=0.2, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        conf = results[0].boxes.conf.cpu().tolist()
        cls_ids = results[0].boxes.cls.cpu().tolist()
        annotated_frame = results[0].plot()

        return boxes, track_ids, conf, cls_ids, annotated_frame

    @staticmethod
    def _save_detection(frame: np.ndarray, box: list, track_id: int, frame_id: int):
        cropped_image = crop_image(frame, box)
        image_path = os.path.join(app_path, str(track_id))
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        filename = os.path.join(image_path, f'{frame_id}.jpg')
        cv2.imwrite(filename, cropped_image)


    def detect_cars_on_road(self):
        # Open the Video File
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            raise ValueError("Error reading video file")

        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, frames_count = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_writer = cv2.VideoWriter(self.annotated_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        for frame_count, _ in enumerate(tqdm(range(frames_count), desc="Processing frames")):
            success, frame = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            boxes, track_ids, conf, cls_ids, annotated_frame = self._perform_detection(frame)

            for track_id, box in zip(track_ids, boxes):
                self._save_detection(frame, box, track_id, frame_count)

            video_writer.write(annotated_frame)

        # Release all Resources:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        print(f'Annotated video location {self.annotated_video_path}')





if __name__ == '__main__':
    app_path = get_app_path('vehicle_detection')
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='input video path')
    args = parser.parse_args()




    v = VehicleDetectionPipeline(args.video_path, app_path)
    v.detect_cars_on_road()



