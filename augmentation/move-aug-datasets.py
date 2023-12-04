import os
import random
import shutil

# Define the original and target folder paths
original_folder = './augmentation/aug_data'
target_folder = './datasets/'

# Define the target subfolder names
subfolders = ['train', 'test', 'valid']

# Define the percentages for each group
percentages = [0.84, 0.15, 0.01]

# Create the target subfolders
for subfolder in subfolders:
    os.makedirs(os.path.join(target_folder, subfolder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_folder, subfolder, 'labels'), exist_ok=True)

# List all files in the original folder
files = os.listdir(os.path.join(original_folder, 'images'))

# Shuffle the files randomly
random.shuffle(files)

# Calculate the number of files for each group
total_files = len(files)
train_count = int(total_files * percentages[0])
valid_count = int(total_files * percentages[1])
test_count = total_files - train_count - valid_count

# Move files to the target folders
for i, file in enumerate(files):
    txt_file = file.replace('.jpg', '.txt')

    source_image_path = os.path.join(original_folder, 'images', file)
    source_label_path = os.path.join(original_folder, 'labels', txt_file)
    
    if i < train_count:
        target_image_path = os.path.join(target_folder, 'train', 'images', file)
        target_label_path = os.path.join(target_folder, 'train', 'labels', txt_file)
    elif i < train_count + valid_count:
        target_image_path = os.path.join(target_folder, 'valid', 'images', file)
        target_label_path = os.path.join(target_folder, 'valid', 'labels', txt_file)
    else:
        target_image_path = os.path.join(target_folder, 'test', 'images', file)
        target_label_path = os.path.join(target_folder, 'test', 'labels', txt_file)
    
    shutil.move(source_image_path, target_image_path)
    shutil.move(source_label_path, target_label_path)

print('Files moved successfully!')