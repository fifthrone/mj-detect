import cv2
import numpy as np
import os
import chardet


# Set the paths for the images and labels folders
images_folder = './augmentation/aug_data/images'
labels_folder = './augmentation/aug_data/labels'

# Get the list of image files and sort them
image_files = sorted(os.listdir(images_folder))




for image_file in image_files[:6]:
    if not image_file.endswith('.jpg'):
      continue  
    
    # Load the image
    image_path = os.path.join(images_folder, image_file)
    image = cv2.imread(image_path)

    # Load the corresponding label file
    label_file = image_file.replace('.jpg', '.txt')
    label_path = os.path.join(labels_folder, label_file)

    # Read the label file and extract bounding box information
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        print(line)
        print(line.strip().split())
        class_id, center_x, center_y, width, height = list(map(float, line.strip().split()))
        
        # Calculate the bounding box corners
        x_min = int((center_x - width / 2) * image.shape[1])
        y_min = int((center_y - height / 2) * image.shape[0])
        x_max = int((center_x + width / 2) * image.shape[1])
        y_max = int((center_y + height / 2) * image.shape[0])
        
        # Draw the bounding box rectangle on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Display the class ID as text
        cv2.putText(image, str(int(class_id)), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow('Image with Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()