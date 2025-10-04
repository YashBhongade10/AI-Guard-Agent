import face_recognition
import os
import pickle
from PIL import Image
import numpy as np
import cv2  # <--- IMPORT OPENCV

# --- Configuration ---
AGENT_DIR = r'D:\Other Files\Games\Agent'
TRUSTED_FACES_DIR = os.path.join(AGENT_DIR, 'trusted_faces')
ENCODINGS_FILE = os.path.join(AGENT_DIR, 'encodings.pkl')

def enroll_faces():
    """
    Processes images in the TRUSTED_FACES_DIR, computes face embeddings,
    and saves them to a pickle file.
    """
    print("Starting face enrollment...")
    known_encodings = []
    known_names = []

    if not os.path.exists(TRUSTED_FACES_DIR):
        print(f"ERROR: Trusted faces directory '{TRUSTED_FACES_DIR}' not found.")
        print("Please create it and add subdirectories for each trusted person with their photos.")
        return

    # Loop over each person in the trusted faces directory
    for person_name in os.listdir(TRUSTED_FACES_DIR):
        person_dir = os.path.join(TRUSTED_FACES_DIR, person_name)

        if not os.path.isdir(person_dir):
            continue

        print(f"Processing images for: {person_name}")
        # Loop over each image of the person
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_dir, filename)
                
                try:
                    # --- NEW ROBUST METHOD ---
                    # 1. Load image with OpenCV
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"  - WARNING: OpenCV could not open {filename}. Skipping.")
                        continue
                        
                    # 2. Convert from BGR (OpenCV default) to RGB (face_recognition default)
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # --- END OF NEW METHOD ---
                    
                    # Find face locations using the RGB image
                    face_locations = face_recognition.face_locations(rgb_image)

                    if len(face_locations) == 1:
                        # Get face encoding from the RGB image
                        face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
                        
                        # Add the encoding and name to our lists
                        known_encodings.append(face_encoding)
                        known_names.append(person_name)
                        print(f"  - Successfully encoded {filename}")
                    elif len(face_locations) == 0:
                        print(f"  - WARNING: No face found in {filename}. Skipping.")
                    else:
                        print(f"  - WARNING: More than one face found in {filename}. Skipping.")

                except Exception as e:
                    print(f"  - ERROR processing {filename}: {e}")

    # Save the encodings to a file
    print("\nSaving encodings...")
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump({'encodings': known_encodings, 'names': known_names}, f)
    
    print(f"Enrollment complete. Encodings saved to '{ENCODINGS_FILE}'")
    
    if len(known_names) > 0:
        print(f"Enrolled {len(known_names)} images for {len(set(known_names))} people.")
    else:
        print("No images were successfully enrolled. Please check your images and directory structure.")

if __name__ == '__main__':
    enroll_faces()

