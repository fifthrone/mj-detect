import os
import cv2
import numpy as np

NOISE = 0.6 #0.4, 0.6

def add_noise(image):
    mean = 0 
    #This sets the mean (average) value of the Gaussian distribution to 0. In the context of image noise, this means that the noise added will, on average, not lighten or darken the image.
    std_dev = NOISE
    #This sets the standard deviation of the Gaussian distribution to 10. The standard deviation is a measure of how spread out the values are in a distribution. In this context, it controls the intensity of the noise: a higher standard deviation will result in more extreme light and dark noise pixels, while a lower standard deviation will make the noise subtler.
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
    #This line generates the actual noise. The np.random.normal function draws random numbers from a Gaussian distribution defined by the specified mean and standard deviation. The third argument, image.shape, specifies the shape of the output, so this will generate a 3D array of noise values (one for each pixel in the image) if the image is colored, or a 2D array if the image is grayscale. Each value in this array will be a random noise value that should be added to the corresponding pixel in the image.
    return cv2.add(image, noise)

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

        # Augmentation: add noise
        noisy_image = add_noise(image.copy())
        cv2.imwrite(os.path.join(aug_image_dir, base + f'_noise{NOISE}.jpg'), noisy_image)
        
        # As the bounding box is not affected, we can just copy the original label file
        os.system(f"cp {label_path} {os.path.join(aug_label_dir, base + f'_noise{NOISE}.txt')}")

if __name__ == '__main__':
    main()