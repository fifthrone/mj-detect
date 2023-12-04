import os
import cv2
import numpy as np
import math

DEGREE = 80

def interpolate(x, x1, x2, y1, y2):
    # Calculate the slope of the line
    m = (y2 - y1) / (x2 - x1)

    # Calculate the y-intercept of the line
    b = y1 - m * x1

    # Return the interpolated value
    return m * x + b

def load_bbox(file_path, image_shape):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        coords = list(map(float, line.strip().split()))
        bboxes.append(coords)
    return bboxes

def save_bbox(file_path, bboxes, image_shape):
    with open(file_path, 'w') as f:
        for bbox in bboxes:
            class_id, center_x, center_y, width, height = bbox
            f.write(f'{int(class_id)} {center_x} {center_y} {width} {height}\n')
            # f.write(' '.join(map(str, bbox)) + '\n')

def rotate_point(point, angle, image_center):
    angle = math.radians(angle)
    offsetX, offsetY = image_center
    cos_val = math.cos(angle)
    sin_val = math.sin(angle)
    x, y = point[0] - offsetX, point[1] - offsetY
    new_x = cos_val * x - sin_val * y + offsetX
    new_y = sin_val * x + cos_val * y + offsetY
    return [new_x, new_y]

def rotate_bbox(bbox, angle, img_width, img_height):
    class_id, cx, cy, bw, bh = bbox
    cx *= img_width
    cy *= img_height
    bw *= img_width
    bh *= img_height
    corners = [(cx - bw/2, cy - bh/2), (cx + bw/2, cy - bh/2), (cx - bw/2, cy + bh/2), (cx + bw/2, cy + bh/2)]
    corners = [rotate_point(corner, angle, (img_width/2, img_height/2)) for corner in corners]
    min_x = min(corners, key=lambda t: t[0])[0]
    min_y = min(corners, key=lambda t: t[1])[1]
    max_x = max(corners, key=lambda t: t[0])[0]
    max_y = max(corners, key=lambda t: t[1])[1]

    new_center_x = (min_x + max_x) / 2 / img_width
    new_center_y = (min_y + max_y) / 2 / img_height
    new_width = (max_x - min_x) / img_width
    new_height = (max_y - min_y) / img_height

    # if abs(angle) > 45:
    #     raise Exception("abs(angle) > 45, pls improve your code")
    # # shrink the bbox a little bit (width and height * 0.8 depend on angle shrink more at 45) [0,45] => [1, 0.8]
    # shrink_amount = interpolate(abs(angle), 0, 45, 1, 0.5)

    shrink_amount = 0.8

    new_width *= shrink_amount
    new_height *= shrink_amount

    new_bbox = [class_id, new_center_x, new_center_y, new_width, new_height]

    return new_bbox

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

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

        # Augmentation: -DEGREE degree rotation
        rotated_image = rotate_image(image, -DEGREE)
        rotated_bboxes = [rotate_bbox(bbox, DEGREE, image.shape[1], image.shape[0]) for bbox in bboxes]
        cv2.imwrite(os.path.join(aug_image_dir, base + f'_rot-{DEGREE}.jpg'), rotated_image)
        save_bbox(os.path.join(aug_label_dir, base + f'_rot-{DEGREE}.txt'), rotated_bboxes, image.shape)

        # Augmentation: DEGREE degree rotation
        rotated_image = rotate_image(image, DEGREE)
        rotated_bboxes = [rotate_bbox(bbox, -DEGREE, image.shape[1], image.shape[0]) for bbox in bboxes]
        cv2.imwrite(os.path.join(aug_image_dir, base + f'_rot{DEGREE}.jpg'), rotated_image)
        save_bbox(os.path.join(aug_label_dir, base + f'_rot{DEGREE}.txt'), rotated_bboxes, image.shape)


if __name__ == '__main__':
    main()