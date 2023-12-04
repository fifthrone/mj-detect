import os
import cv2
import numpy as np


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

def rotate_bbox(image, bboxes, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)

    rotated_bboxes = []
    for bbox in bboxes:
        class_id, xmin, ymin, xmax, ymax = bbox
        corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        corners = np.hstack((corners, np.ones((4, 1))))
        corners = np.dot(M, corners.T).T
        xmin, ymin = np.min(corners, axis=0)[:2]
        xmax, ymax = np.max(corners, axis=0)[:2]
        rotated_bboxes.append([class_id, xmin, ymin, xmax, ymax])

    return rotated_bboxes


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

        # Augmentation: clockwise 90 degrees rotation
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_bboxes = rotate_bbox(image, bboxes, -90)
        cv2.imwrite(os.path.join(aug_image_dir, base + '_cw90.jpg'), rotated_image)
        save_bbox(os.path.join(aug_label_dir, base + '_cw90.txt'), rotated_bboxes, image.shape)

        # Augmentation: counter-clockwise 90 degrees rotation
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_bboxes = rotate_bbox(image, bboxes, 90)
        cv2.imwrite(os.path.join(aug_image_dir, base + '_ccw90.jpg'), rotated_image)
        save_bbox(os.path.join(aug_label_dir, base + '_ccw90.txt'), rotated_bboxes, image.shape)

        # Augmentation: counter-clockwise 180 degrees rotation
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        rotated_bboxes = rotate_bbox(image, bboxes, 180)
        cv2.imwrite(os.path.join(aug_image_dir, base + '_ccw180.jpg'), rotated_image)
        save_bbox(os.path.join(aug_label_dir, base + '_ccw180.txt'), rotated_bboxes, image.shape)


if __name__ == '__main__':
    main()