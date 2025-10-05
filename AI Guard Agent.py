import speech_recognition as sr
import cv2
import face_recognition
import pickle
import numpy as np
import threading
import time
import os
from gtts import gTTS
from playsound import playsound
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in the .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# --- Global State ---
guard_mode_active = threading.Event()
stop_threads = False
last_unrecognized_time = None
unrecognized_face_id_counter = 0
unrecognized_faces_memory = {}

# --- Constants & Paths ---
AGENT_DIR = r'D:\Other Files\Games\Agent'
ENCODINGS_FILE = os.path.join(AGENT_DIR, 'encodings.pkl')
ACTIVATION_COMMAND = "activate protection"
DEACTIVATION_COMMAND = "deactivate"
UNRECOGNIZED_COOLDOWN = 15
ESCALATION_LEVELS = [
    (5, "You are an AI room guard. An unrecognized person has been detected. Generate a polite but firm Level 1 spoken response, asking who they are and their purpose."),
    (15, "The same unrecognized person is still present. Generate a more assertive Level 2 warning. State that this is a restricted area and they should leave if they are not authorized."),
    (30, "The intruder has not left. Generate a stern and final Level 3 warning. Announce that security protocols are being initiated and authorities will be alerted if they do not leave immediately.")
]
CAMERA_INDEX = 0

# --- Helper Functions ---
def speak(text, filename="response.mp3"):
    """Converts text to speech and plays it."""
    try:
        print(f"AI GUARD: {text}")
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print(f"Error in TTS or audio playback: {e}")

def get_llm_response(prompt):
    """Gets a response from the Gemini LLM."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        return "I have detected an unauthorized presence. Please leave the area immediately."

# --- Core Logic ---
def listen_for_commands():
    """Waits for user to press Enter, then listens for a command."""
    global stop_threads
    recognizer = sr.Recognizer()
    
    print("Command listener started. Press Enter in this terminal to activate the microphone.")
    
    while not stop_threads:
        try:
            # This line will pause the script and wait for the user to press Enter.
            input(">>> Press Enter to speak a command...")

            # If the stop signal was sent while waiting for input, exit the loop.
            if stop_threads:
                break

            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1) 
                print("Listening now...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            
            text = recognizer.recognize_google(audio).lower()
            print(f"Heard: '{text}'")

            if ACTIVATION_COMMAND in text and not guard_mode_active.is_set():
                print("Activation command received.")
                speak("Guard mode activated.")
                guard_mode_active.set()
            elif DEACTIVATION_COMMAND in text and guard_mode_active.is_set():
                print("Deactivation command received.")
                speak("Guard mode deactivated.")
                guard_mode_active.clear()
        
        except sr.WaitTimeoutError:
            print("No command heard. Press Enter to try again.")
            continue
        except sr.UnknownValueError:
            print("Could not understand the audio. Press Enter to try again.")
            continue
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            # This handles the case where the program is closing while waiting for input()
            if not stop_threads:
                 print(f"An error occurred in command listener: {e}")
            break

def monitor_faces():
    """Monitors webcam for faces when guard mode is active."""
    global stop_threads, unrecognized_faces_memory

    print("Face monitor started.")
    
    try:
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
        known_face_encodings = data['encodings']
        known_face_names = data['names']
    except FileNotFoundError:
        print(f"ERROR: Encodings file '{ENCODINGS_FILE}' not found. Please run enroll_faces.py first.")
        stop_threads = True
        return

    video_capture = None

    while not stop_threads:
        if not guard_mode_active.is_set():
            if video_capture and video_capture.isOpened():
                video_capture.release()
                cv2.destroyAllWindows()
                video_capture = None
            time.sleep(1)
            continue

        if video_capture is None:
            video_capture = cv2.VideoCapture(CAMERA_INDEX)
            if not video_capture.isOpened():
                print(f"FATAL ERROR: Could not open webcam at index {CAMERA_INDEX}.")
                speak("Error: Cannot access the webcam.")
                video_capture = None
                guard_mode_active.clear()
                time.sleep(5)
                continue
            print("Webcam successfully opened.")

        ret, frame = video_capture.read()
        if not ret:
            print("Warning: Could not read frame from camera.")
            continue

        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unrecognized"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            
            if name == "Unrecognized":
                handle_unrecognized_face(face_encoding)
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('AI Guard', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            break
    
    if video_capture and video_capture.isOpened():
        video_capture.release()
    cv2.destroyAllWindows()
    print("Face monitor stopped.")


def handle_unrecognized_face(current_encoding):
    """Manages the escalation logic for a single unrecognized face."""
    global unrecognized_faces_memory, unrecognized_face_id_counter
    
    now = time.time()
    matched_id = None
    
    for face_id, data in unrecognized_faces_memory.items():
        distance = face_recognition.face_distance([data['encoding']], current_encoding)
        if distance[0] < 0.6:
            matched_id = face_id
            break

    if matched_id is not None:
        unrecognized_faces_memory[matched_id]['last_seen'] = now
        data = unrecognized_faces_memory[matched_id]
        
        time_since_first_seen = now - data['first_seen']
        current_level = data['escalation_level']

        if current_level < len(ESCALATION_LEVELS):
            delay, _ = ESCALATION_LEVELS[current_level]
            if time_since_first_seen > delay and (now - data.get('last_spoken_time', 0) > UNRECOGNIZED_COOLDOWN):
                _, prompt = ESCALATION_LEVELS[current_level]
                response = get_llm_response(prompt)
                threading.Thread(target=speak, args=(response,)).start()
                unrecognized_faces_memory[matched_id]['escalation_level'] += 1
                unrecognized_faces_memory[matched_id]['last_spoken_time'] = now
    else:
        unrecognized_face_id_counter += 1
        new_id = unrecognized_face_id_counter
        unrecognized_faces_memory[new_id] = {
            'encoding': current_encoding, 'first_seen': now, 'last_seen': now,
            'escalation_level': 0, 'last_spoken_time': 0
        }
        
        _, prompt = ESCALATION_LEVELS[0]
        response = get_llm_response(prompt)
        threading.Thread(target=speak, args=(response,)).start()
        unrecognized_faces_memory[new_id]['escalation_level'] += 1
        unrecognized_faces_memory[new_id]['last_spoken_time'] = now


def cleanup_memory():
    """Periodically remove old unrecognized faces from memory."""
    global unrecognized_faces_memory
    while not stop_threads:
        now = time.time()
        expired_ids = [
            face_id for face_id, data in unrecognized_faces_memory.items()
            if now - data['last_seen'] > 300 # 5 minutes
        ]
        for face_id in expired_ids:
            del unrecognized_faces_memory[face_id]
        time.sleep(60)


if __name__ == '__main__':
    command_thread = threading.Thread(target=listen_for_commands)
    monitor_thread = threading.Thread(target=monitor_faces)
    memory_thread = threading.Thread(target=cleanup_memory)

    try:
        command_thread.start()
        monitor_thread.start()
        memory_thread.start()
        
        while command_thread.is_alive() and monitor_thread.is_alive():
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] Shutdown signal received. Closing threads...")
    finally:
        stop_threads = True
        guard_mode_active.clear()
        
        # This helps the input() in the command listener to unblock
        print("Press Enter to allow shutdown to complete...")

        command_thread.join()
        monitor_thread.join()
        memory_thread.join()
        print("[INFO] All threads closed. Shutdown complete.")


