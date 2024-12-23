import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Base folder path where subfolders s1 to s7 are stored
base_folder = r'Filepath_of_preprocessed_frame'

# List of folder names to be processed (s1 to s7)
subject_folders = [f's{i}' for i in range(1, 8)]  # ['s1', 's2', ..., 's7']

# Subfolders inside each subject folder (T1, T2, T3)
video_folders = ['T1', 'T2', 'T3']

def extract_ycgcr_traces(images_path):
    y_values = []
    cg_values = []
    cr_values = []

    # Loop through each image in the folder
    for image_file in sorted(os.listdir(images_path)):
        if image_file.endswith('.png'):
            image_path = os.path.join(images_path, image_file)

            # Read the image
            image = cv2.imread(image_path)

            # Check if image was successfully loaded
            if image is not None:
                # Convert the image to YCgCr color space
                ycgcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

                # Split the image into Y, Cg, and Cr channels
                y, cg, cr = cv2.split(ycgcr_image)

                # Create masks to exclude zero-intensity pixels
                mask_y = y > 0
                mask_cg = cg > 0
                mask_cr = cr > 0

                # Calculate the mean intensity only for non-zero pixels
                y_mean = np.mean(y[mask_y]) if np.any(mask_y) else 0
                cg_mean = np.mean(cg[mask_cg]) if np.any(mask_cg) else 0
                cr_mean = np.mean(cr[mask_cr]) if np.any(mask_cr) else 0

                y_values.append(y_mean)
                cg_values.append(cg_mean)
                cr_values.append(cr_mean)

    return np.array(y_values), np.array(cg_values), np.array(cr_values)

# Iterate over the subject folders (s1 to s7)
for subject_folder in subject_folders:
    subject_path = os.path.join(base_folder, subject_folder)
    
    # Check if the subject folder exists (for safety)
    if not os.path.exists(subject_path):
        print(f"Subject folder {subject_path} does not exist. Skipping...")
        continue
    
    # Iterate over the video folders (T1, T2, T3) inside each subject folder
    for video_folder in video_folders:
        video_path = os.path.join(subject_path, video_folder)
        
        # Check if the video folder exists
        if not os.path.exists(video_path):
            print(f"Video folder {video_path} does not exist. Skipping...")
            continue
        
        # Extract YCgCr traces for images in this folder
        y_values, cg_values, cr_values = extract_ycgcr_traces(video_path)
        
        # Plot YCgCr intensities for this video folder
        plt.figure(figsize=(10, 6))
        frames = range(len(y_values))  # x-axis as the frame number

        plt.plot(frames, y_values, color='y', label='Y (Luminance) Intensity')
        plt.plot(frames, cg_values, color='g', label='Cg Intensity')
        plt.plot(frames, cr_values, color='r', label='Cr Intensity')

        plt.title(f'YCgCr Color Traces for {subject_folder} - {video_folder}')
        plt.xlabel('Frame Number')
        plt.ylabel('Color Intensity')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()
