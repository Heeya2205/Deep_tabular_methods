import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Base folder path where subfolders s1 to s7 are stored
base_folder = r'.\preprocessed_images'
subject_folders = [f's{i}' for i in range(1, 57)]  # ['s1', 's2', ..., 's57']
video_folders = ['T1', 'T2', 'T3']

def extract_ycgcr_traces(images_path):
    y_values = []
    cg_values = []
    cr_values = []
    
    for image_file in sorted(os.listdir(images_path)):
        if image_file.endswith('.png'):
            image_path = os.path.join(images_path, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                ycgcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                y, cg, cr = cv2.split(ycgcr_image)
                mask_y = y > 0
                mask_cg = cg > 0
                mask_cr = cr > 0
                y_mean = np.mean(y[mask_y]) if np.any(mask_y) else 0
                cg_mean = np.mean(cg[mask_cg]) if np.any(mask_cg) else 0
                cr_mean = np.mean(cr[mask_cr]) if np.any(mask_cr) else 0
                y_values.append(y_mean)
                cg_values.append(cg_mean)
                cr_values.append(cr_mean)
                
    return np.array(y_values), np.array(cg_values), np.array(cr_values)

def butterworth_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def pos_method(y_values, cg_values, cr_values, fs=30):
    color_signal_1 = y_values - cg_values
    color_signal_2 = cr_values - (y_values + cg_values) / 2
    mean_1 = np.mean(color_signal_1)
    mean_2 = np.mean(color_signal_2)
    signal_1 = color_signal_1 - mean_1
    signal_2 = color_signal_2 - mean_2
    pos_signal = signal_1 + signal_2
    filtered_signal = butterworth_filter(pos_signal, lowcut=0.7, highcut=2.5, fs=fs, order=4)
    return filtered_signal

# Function to compute heart rate and HRV
def compute_heart_rate_and_hrv(peaks, fs):
    # Calculate the time intervals between peaks in seconds
    intervals = np.diff(peaks) / fs  # Convert frame counts to seconds
    if len(intervals) == 0:
        return None, None  # No intervals available
    
    # Heart rate (in beats per minute)
    heart_rate = 60 / np.mean(intervals) if np.mean(intervals) > 0 else 0
    
    # Heart Rate Variability (HRV)
    hrv = np.std(intervals) if len(intervals) > 1 else 0
    
    return heart_rate, hrv

# Iterate over the subject folders (s1 to s7)
for subject_folder in subject_folders:
    subject_path = os.path.join(base_folder, subject_folder)
    
    if not os.path.exists(subject_path):
        print(f"Subject folder {subject_path} does not exist. Skipping...")
        continue
    
    for video_folder in video_folders:
        video_path = os.path.join(subject_path, video_folder)
        
        if not os.path.exists(video_path):
            print(f"Video folder {video_path} does not exist. Skipping...")
            continue
        
        y_values, cg_values, cr_values = extract_ycgcr_traces(video_path)
        rppg_signal = pos_method(y_values, cg_values, cr_values, fs=30)
        
        # Step 4: Detect peaks in the rPPG signal
        peaks, _ = find_peaks(rppg_signal, height=0)  # You may want to adjust the height threshold

        # Step 5: Compute heart rate and HRV
        heart_rate, hrv = compute_heart_rate_and_hrv(peaks, fs=30)

        # Plot the filtered rPPG signal and the detected peaks
        plt.figure(figsize=(10, 6))
        frames = range(len(rppg_signal))
        plt.plot(frames, rppg_signal, label='rPPG Signal (Butterworth)', color='b')
        plt.plot(peaks, rppg_signal[peaks], "x", label='Detected Peaks', color='r')
        plt.title(f'rPPG Signal for {subject_folder} - {video_folder} (POS Method with Butterworth Filter)')
        plt.xlabel('Frame Number')
        plt.ylabel('Signal Intensity')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()
        
        # Print the heart rate and HRV
        print(f'Heart Rate for {subject_folder} - {video_folder}: {heart_rate:.2f} bpm')
        print(f'Heart Rate Variability (HRV) for {subject_folder} - {video_folder}: {hrv:.4f} seconds')
