import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Load the Haar Cascade face detector
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Base path to the main folder containing subfolders (subject folders)
base_path = r'File_path_name'

# Main output folder to store the landmarks
main_output_folder = r'directory_name\To store output '

# Create the main output folder if it doesn't exist
if not os.path.exists(main_output_folder):
    os.makedirs(main_output_folder)

# List of video names to be processed (T1, T2, T3)
video_files = ['T1.avi', 'T2.avi', 'T3.avi']

# Process subjects from s1 to s56
for subject_id in range(1, 57):
    # Define the subject folder (e.g., s1, s2, ..., s56)
    subject_folder = os.path.join(base_path, f's{subject_id}')
    
    # Check if subject folder exists (to avoid errors if a folder is missing)
    if not os.path.exists(subject_folder):
        print(f"Folder {subject_folder} does not exist. Skipping...")
        continue

    # Create a subject-specific folder in the face_detection output folder
    subject_output_folder = os.path.join(main_output_folder, f's{subject_id}')
    
    # Create the subject output folder if it doesn't exist
    if not os.path.exists(subject_output_folder):
        os.makedirs(subject_output_folder)

    # Process each video file (T1.avi, T2.avi, T3.avi) for the current subject
    for video_name in video_files:
        # Construct the full video path based on the subject id and video type (T1, T2, T3)
        video_path = os.path.join(subject_folder, f'vid_s{subject_id}_{video_name}')
        
        # Check if the video file exists
        if not os.path.exists(video_path):
            print(f"Video {video_path} does not exist. Skipping...")
            continue
        
        # Create a folder for each video (T1, T2, T3) under the subject folder
        video_output_folder = os.path.join(subject_output_folder, video_name[:-4])  # Remove ".avi"
        
        # Create the video-specific output folder if it doesn't exist
        if not os.path.exists(video_output_folder):
            os.makedirs(video_output_folder)
        
        # Load the video file
        cap = cv2.VideoCapture(video_path)

        # Get the actual frame rate of the video
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        target_fps = 30
        frame_interval = 1.0 / target_fps  # Time interval between frames to extract (1/30 seconds)
        video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps  # Total duration of the video in seconds
        expected_frames = int(target_fps * video_duration)  # Expected number of frames to be processed

        frame_count = 0
        processed_count = 0
        time_elapsed = 0.0

        # Process frames from the video
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break  # End of video

            # Calculate the current timestamp of the video
            current_time = frame_count / video_fps

            # Only process frames that match the desired frame interval (30 fps)
            if current_time >= time_elapsed:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces using Haar Cascade
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # If a face is detected, process the largest bounding box
                if len(faces) > 0:
                    x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]

                    # Crop the detected face region
                    face_region = frame[y:y+h, x:x+w]

                    # Convert to RGB for MediaPipe processing
                    face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

                    # Detect landmarks in the face region using MediaPipe
                    result = face_mesh.process(face_rgb)

                    if result.multi_face_landmarks:
                        for face_landmarks in result.multi_face_landmarks:
                            # Extract the landmark points
                            landmark_points = []
                            for landmark in face_landmarks.landmark:
                                landmark_x = int(landmark.x * w)
                                landmark_y = int(landmark.y * h)
                                landmark_points.append([landmark_x, landmark_y])

                            landmark_points = np.array(landmark_points)

                            # Create a convex hull from the landmarks
                            hull = cv2.convexHull(landmark_points)

                            # Create a mask for the face region
                            mask = np.zeros(face_region.shape[:2], dtype=np.uint8)
                            cv2.fillConvexPoly(mask, hull, 255)

                            # Apply the mask to the face region
                            masked_face = cv2.bitwise_and(face_region, face_region, mask=mask)

                            # Save the masked face image
                            output_img_path = os.path.join(video_output_folder, f'landmark_frame_{processed_count + 1}.png')
                            cv2.imwrite(output_img_path, masked_face)
                            print(f"Saved processed image: {output_img_path}")

                        # Increment the processed frame count
                        processed_count += 1

                # Update the time to the next expected frame interval
                time_elapsed += frame_interval

            # Increment the frame count
            frame_count += 1

        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()
