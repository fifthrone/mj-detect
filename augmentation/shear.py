import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import affine_transform

def load_bbox(file_path, image_shape):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        class_id, center_x, center_y, width, height = list(map(float, line.strip().split()))
        x1 = (center_x - width / 2) * image_shape[1] #top left x
        y1 = (center_y - height / 2) * image_shape[0] #top left y
        x2 = (center_x + width / 2) * image_shape[1] #bottom right x
        y2 = (center_y + height / 2) * image_shape[0] #bottom right y
        bboxes.append([class_id, x1, y1, x2, y2])
    return bboxes


def save_bbox(file_path, bboxes, image_shape):
    with open(file_path, 'w') as f:
        for bbox in bboxes:
            class_id, x1, y1, x2, y2 = bbox
            center_x = ((x2 + x1) / 2) / image_shape[1]
            center_y = ((y2 + y1) / 2) / image_shape[0]
            width = (x2 - x1) / image_shape[1]
            height = (y2 - y1) / image_shape[0]
            f.write(f'{int(class_id)} {center_x} {center_y} {width} {height}\n')


def shear_bbox(bboxes, mx, my):
    sheared_bboxes = []
    for bbox in bboxes:
        class_id, x1, y1, x2, y2 = bbox
        # Define the original corners
        top_left = np.array([x1, y1])
        bottom_right = np.array([x2, y2])
        top_right = np.array([x2, y1])
        bottom_left = np.array([x1, y2])

        # Apply shear transformation to each corner
        sheared_top_left = top_left + np.array([mx * top_left[1], my * top_left[0]])
        sheared_bottom_right = bottom_right + np.array([mx * bottom_right[1], my * bottom_right[0]])
        sheared_top_right = top_right + np.array([mx * top_right[1], my * top_right[0]])
        sheared_bottom_left = bottom_left + np.array([mx * bottom_left[1], my * bottom_left[0]])

        # Compute the new coordinates as the average of the sheared coordinates of two opposite corners
        new_x1 = (sheared_top_left[0] + sheared_bottom_left[0]) / 2
        new_y1 = (sheared_top_left[1] + sheared_top_right[1]) / 2
        new_x2 = (sheared_bottom_right[0] + sheared_top_right[0]) / 2
        new_y2 = (sheared_bottom_right[1] + sheared_bottom_left[1]) / 2

        # Append the new bounding box to the list
        sheared_bboxes.append([class_id, new_x1, new_y1, new_x2, new_y2])

    return sheared_bboxes

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

        # Augmentation: shear
        mx = np.random.uniform(-0.267, 0.267)  # tan(15 degrees) ≈ 0.267
        my = np.random.uniform(-0.267, 0.267)  # tan(15 degrees) ≈ 0.267
        M = np.float32([[1, mx, 0], [my, 1, 0]])
        sheared_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        sheared_bboxes = shear_bbox(bboxes, mx, my)
        cv2.imwrite(os.path.join(aug_image_dir, base + '_shear.jpg'), sheared_image)
        save_bbox(os.path.join(aug_label_dir, base + '_shear.txt'), sheared_bboxes, image.shape)



if __name__ == '__main__':
    main()