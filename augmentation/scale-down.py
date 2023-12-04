import os
import cv2
import numpy as np
import math

SCALE = 0.9

def load_bbox(file_path, image_shape):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        # Assuming the bounding box is in the format: class_id, center_x, center_y, width, height
        values = list(map(float, line.strip().split()))
        class_id, center_x, center_y, width, height = values
        # Convert normalized coordinates to pixel coordinates
        center_x *= image_shape[1]
        center_y *= image_shape[0]
        width *= image_shape[1]
        height *= image_shape[0]
        xmin = center_x - width / 2
        ymin = center_y - height / 2
        xmax = center_x + width / 2
        ymax = center_y + height / 2
        bboxes.append([class_id, xmin, ymin, xmax, ymax])
    return bboxes

def save_bbox(file_path, bboxes, image_shape):
    with open(file_path, 'w') as f:
        for bbox in bboxes:
            # Convert pixel coordinates back to normalized coordinates
            class_id, xmin, ymin, xmax, ymax = bbox
            center_x = (xmin + xmax) / 2 / image_shape[1]
            center_y = (ymin + ymax) / 2 / image_shape[0]
            width = (xmax - xmin) / image_shape[1]
            height = (ymax - ymin) / image_shape[0]
            f.write(f'{int(class_id)} {center_x} {center_y} {width} {height}\n')


def scale_image(image, scale):
    h, w = image.shape[:2]
    scaled_h, scaled_w = int(h * scale), int(w * scale)

    # Scale down the image
    scaled_image = cv2.resize(image, (scaled_w, scaled_h))

    # Create a new image of the original size with black background
    new_image = np.zeros_like(image)

    # Compute the starting points to center the scaled image
    start_h, start_w = (h - scaled_h) // 2, (w - scaled_w) // 2

    # Place the scaled image on the black background
    new_image[start_h:start_h+scaled_h, start_w:start_w+scaled_w] = scaled_image

    return new_image, (start_h, start_w)

def scale_bbox(bbox, start_point, scale):
    class_id, xmin, ymin, xmax, ymax = bbox
    start_h, start_w = start_point

    # Scale and shift the bounding box coordinates
    new_bbox = [
        class_id,
        int(xmin * scale) + start_w,
        int(ymin * scale) + start_h,
        int(xmax * scale) + start_w,
        int(ymax * scale) + start_h
    ]

    return new_bbox

def main():
    image_dir = './augmentation/images'
    label_dir = './augmentation/labels'
    aug_image_dir = './augmentation/aug_data/images'
    aug_label_dir = './augmentation/aug_data/labels'
    os.makedirs(aug_image_dir, exist_ok=True)
    os.makedirs(aug_label_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if not filename.endswith('.jpg'):
            continue

        base = os.path.splitext(filename)[0]
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, base + '.txt')

        image = cv2.imread(image_path)
        bboxes = load_bbox(label_path, image.shape)

        # Augmentation: scale down by SCALE and add black edges
        scaled_image, start_point = scale_image(image, SCALE)
        scaled_bboxes = [scale_bbox(bbox, start_point, SCALE) for bbox in bboxes]
        cv2.imwrite(os.path.join(aug_image_dir, base + f'_scaled{SCALE}.jpg'), scaled_image)
        save_bbox(os.path.join(aug_label_dir, base + f'_scaled{SCALE}.txt'), scaled_bboxes, image.shape)


if __name__ == '__main__':
    main()