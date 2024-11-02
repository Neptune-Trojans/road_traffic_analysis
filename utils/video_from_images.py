import os
import cv2

from utils.image_utils import get_images_from_path


def create_video_from_images(image_folder: str, output_video_path: str, frame_size=(640, 480), fps=30):
    # Get a list of image file paths
    image_files = get_images_from_path(image_folder)
    image_files.sort()  # Optional: sort files if they need to be in a specific order

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Specify codec for video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        # Resize the image
        resized_image = cv2.resize(image, frame_size)

        # Write the resized image to the video
        video_writer.write(resized_image)

    # Release the VideoWriter object
    video_writer.release()
    print(f"Video saved to {output_video_path}")

if __name__ == '__main__':
    image_folder = '/Users/yudkin/.flamingo/Applications/plate_detection'
    output_video_path = os.path.join(image_folder, 'license_pate.mp4')
    create_video_from_images(image_folder, output_video_path, fps=10)
