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

# Configure the Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# --- Global State ---
guard_mode_active = threading.Event()
stop_threads = False
last_unrecognized_time = None
unrecognized_face_id_counter = 0
unrecognized_faces_memory = {} # Stores info about unrecognized faces

# --- Constants ---
ACTIVATION_COMMAND = "activate protection"
DEACTIVATION_COMMAND = "stop protection"
ENCODINGS_FILE = 'encodings.pkl'
UNRECOGNIZED_COOLDOWN = 15  # seconds before re-engaging the same face
ESCALATION_LEVELS = [
    (5, "You are an AI room guard. An unrecognized person has been detected. Generate a firm Level 1 spoken response, asking who they are and their purpose in hindi language one line dont translate and ask him to get out."),
    (10, "The same unrecognized person is still present. Generate a more assertive Level 2 warning in hindi language one line dont translate. State that this is a restricted area and they should leave if they are not authorized."),
    (15, "The intruder has not left. Generate a stern and final Level 3 warning in hindi language one line dont translate. Announce that security protocols are being initiated and authorities will be alerted if they do not leave immediately.")
]

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
    """Listens for activation/deactivation commands in a separate thread."""
    global stop_threads
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    print("Command listener started. Say 'activate protection' to activate.")

    while not stop_threads:
        try:
            with microphone as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            
            text = recognizer.recognize_google(audio).lower()
            print(f"Heard: '{text}'")

            if ACTIVATION_COMMAND in text and not guard_mode_active.is_set():
                print("Activation command received.")
                speak("protection mode activated.")
                guard_mode_active.set()
            elif DEACTIVATION_COMMAND in text and guard_mode_active.is_set():
                print("Deactivation command received.")
                speak("protection mode deactivated.")
                guard_mode_active.clear()
        
        except sr.WaitTimeoutError:
            continue # It's normal to have silence
        except sr.UnknownValueError:
            continue # Could not understand audio
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            print(f"An error occurred in command listener: {e}")

def monitor_faces():
    """Monitors webcam for faces when guard mode is active."""
    global stop_threads, unrecognized_faces_memory

    print("Face monitor started.")
    
    # Load known face encodings
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
            video_capture = cv2.VideoCapture(0)
            if not video_capture.isOpened():
                print("Error: Could not open webcam.")
                time.sleep(2)
                continue

        ret, frame = video_capture.read()
        if not ret:
            continue

        # Find all the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        current_person_name = "Unrecognized"
        unrecognized_detected = False

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unrecognized"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            
            current_person_name = name
            if name == "Unrecognized":
                unrecognized_detected = True
                handle_unrecognized_face(face_encoding)

        # Draw boxes and names on the frame
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, current_person_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('AI Guard', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
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
    
    # Check if this face is similar to one we've seen recently
    for face_id, data in unrecognized_faces_memory.items():
        distance = face_recognition.face_distance([data['encoding']], current_encoding)
        if distance[0] < 0.6: # Tolerance to consider it the same person
            matched_id = face_id
            break

    if matched_id is not None:
        # We've seen this person before
        unrecognized_faces_memory[matched_id]['last_seen'] = now
        data = unrecognized_faces_memory[matched_id]
        
        # Check if it's time to escalate
        time_since_first_seen = now - data['first_seen']
        current_level = data['escalation_level']

        if current_level < len(ESCALATION_LEVELS):
            delay, _ = ESCALATION_LEVELS[current_level]
            if time_since_first_seen > delay and (now - data.get('last_spoken_time', 0) > UNRECOGNIZED_COOLDOWN):
                _, prompt = ESCALATION_LEVELS[current_level]
                response = get_llm_response(prompt)
                speak(response)
                unrecognized_faces_memory[matched_id]['escalation_level'] += 1
                unrecognized_faces_memory[matched_id]['last_spoken_time'] = now

    else:
        # This is a new unrecognized person
        unrecognized_face_id_counter += 1
        new_id = unrecognized_face_id_counter
        unrecognized_faces_memory[new_id] = {
            'encoding': current_encoding,
            'first_seen': now,
            'last_seen': now,
            'escalation_level': 0,
            'last_spoken_time': 0
        }

def cleanup_memory():
    """Periodically remove old unrecognized faces from memory."""
    global unrecognized_faces_memory
    while not stop_threads:
        now = time.time()
        # Remove faces not seen for 5 minutes (300 seconds)
        expired_ids = [
            face_id for face_id, data in unrecognized_faces_memory.items()
            if now - data['last_seen'] > 300
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

        command_thread.join()
        monitor_thread.join()
        memory_thread.join()
    except KeyboardInterrupt:
        print("\nShutting down AI Guard...")
        stop_threads = True
        guard_mode_active.clear()
        command_thread.join()
        monitor_thread.join()
        memory_thread.join()
        print("Shutdown complete.")
