import tkinter as tk
import numpy as np
import tensorflow as tf
import librosa
from tkinter import scrolledtext
import threading
import speech_recognition as sr
import pyttsx3
import subprocess
import pywhatkit
import wikipedia
import wolframalpha

# Initialize the recognizer
r = sr.Recognizer()
SAMPLING_RATE = 16000

# Function to convert text to speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

# Function to preprocess live audio data
def preprocess_audio(audio_data):
    # Convert audio data to floating-point format
    audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
    
    # Resample audio if necessary
    if len(audio_data) != SAMPLING_RATE:
        audio_data = librosa.resample(audio_data, orig_sr=len(audio_data), target_sr=SAMPLING_RATE)
    
    # Trim or pad audio to 8000 samples
    target_length = 8000
    if len(audio_data) > target_length:
        audio_data = audio_data[:target_length]
    elif len(audio_data) < target_length:
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')
    
    # Reshape audio data
    audio_data = np.reshape(audio_data, (-1, 8000, 1))
    
    return audio_data

# Function to handle speech recognition and main functionality
def recognize_speech(output_text):
    try:
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.1)
            output_text.delete(1.0, tk.END)
            output_text.insert(tk.END, "Listening...\n")
            audio2 = r.listen(source2)

            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            if 'open' in MyText and 'browser' in MyText:
                output_text.delete(1.0, tk.END)
                output_text.insert(tk.END, "Opening browser...\n")
                SpeakText("Opening the browser")
                program = "C:/Program Files/Mozilla Firefox/firefox.exe"
                subprocess.Popen([program])

            elif 'play' in MyText:
                output_text.delete(1.0, tk.END)
                output_text.insert(tk.END, "Opening YouTube...\n")
                SpeakText("Opening YouTube")
                pywhatkit.playonyt(MyText)
            
            elif 'send ' in MyText and 'message' in MyText:
                output_text.delete(1.0, tk.END)
                output_text.insert(tk.END, "Listening for phone number...\n")

                while True:
                    audio2 = r.listen(source2)
                    phone_no = r.recognize_google(audio2)

                    output_text.insert(tk.END, f"Phone number recognized: {phone_no}\n")
                    output_text.insert(tk.END, "Are you sure about the number?\n")

                    audio2 = r.listen(source2)
                    response = r.recognize_google(audio2)

                    if 'yes' in response:
                        output_text.insert(tk.END, "Enter the message you want to send: \n")
                        audio2 = r.listen(source2)
                        message = r.recognize_google(audio2)
                        output_text.insert(tk.END, f"Your message: {message}\n")
                        pywhatkit.sendwhatmsg_instantly('+' + phone_no, message, wait_time=7)
                        output_text.insert(tk.END, "Message sent successfully!\n")
                        break
                    elif 'no' in response:
                        output_text.delete(1.0, tk.END)
                        continue

            elif 'stop' in MyText:
                print("Stopping the program...")
                quit()

            else:
                appId = 'APER4E-58XJGHAVAK'
                client = wolframalpha.Client(appId)
                res = client.query(MyText)
                output = next(res.results).text

                output_text.delete(1.0, tk.END)
                output_text.insert(tk.END, f"Input question:\n{MyText}\n")
                output_text.insert(tk.END, f"Output response:\n{output}\n")
                SpeakText(output)

    except StopIteration as e:
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, wikipedia.summary(MyText, sentences=2))
        
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
        output_text.insert(tk.END, f"Error: {e}\n")

    except sr.RequestError as e:
        output_text.insert(tk.END, f"Could not request results; {e}\n")

    except sr.UnknownValueError:
        output_text.insert(tk.END, "unknown error occurred\n")

def verify_speaker(output_text):
    try:
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.1)
            output_text.delete(1.0, tk.END)
            output_text.insert(tk.END, "Listening for verification...\n")
            audio2 = r.listen(source2)

            # Speaker Verification
            speakers = ["Benajmin_Netanayu", "Jens_Stoltenberg", "Julia_Gillard", "Margaret_Tarcher", "Nelson_Mandela", "Stefanos_Sauciuc"]

            # Convert the recorded audio to numpy array
            audio_data = np.frombuffer(audio2.frame_data, dtype=np.int16)

            # Preprocess the audio data
            preprocessed_audio = preprocess_audio(audio_data)

            model = tf.keras.models.load_model('model.keras')

            # Perform prediction using the model
            predictions = model.predict(preprocessed_audio)

            # Get the predicted speaker
            predicted_speaker = speakers[np.argmax(predictions)]

            output_text.insert(tk.END, f"Predicted speaker: {predicted_speaker}\n")

            if predicted_speaker == "Stefanos_Sauciuc":
                # Check if the person also said "Hello"
                if "hello" in r.recognize_google(audio2).lower():
                    output_text.insert(tk.END, "Verification complete.\n")
                    verify_button.config(text="Ask Question", command=lambda: threading.Thread(target=start_listening, args=(output_text,)).start())
                    verified_label.config(text="Verified", fg="green")
            else:
                    output_text.insert(tk.END, "Access Denied: Speaker not recognized as Stefanos Sauciuc\n")
                    verified_label.config(text="Access Denied", fg="red")

    except sr.RequestError as e:
        output_text.insert(tk.END, f"Could not request results; {e}\n")

    except sr.UnknownValueError:
        output_text.insert(tk.END, "unknown error occurred\n")

# Function to start the verification process
def start_verification(output_text):
    threading.Thread(target=verify_speaker, args=(output_text,)).start()

# Function to start listening for questions
def start_listening(output_text):
    threading.Thread(target=recognize_speech, args=(output_text,)).start()

# Function to stop speech recognition loop
def stop_listening():
    global running
    running = False
    output_text.delete(1.0, tk.END)

# Create GUI
root = tk.Tk()
root.title("Voice Assistant")

# Create output text area
output_text = scrolledtext.ScrolledText(root, width=60, height=20, font=("Arial", 15))
output_text.pack(padx=10, pady=10)

# Initial message indicating verification is required
output_text.insert(tk.END, "Verification required.\n")

# Create Verify button
verify_button = tk.Button(root, text="Verify", justify="center", width=20, height=2, command=lambda: threading.Thread(target=start_verification, args=(output_text,)).start())
verify_button.pack(side=tk.LEFT, padx=10, pady=10)

# Create Reset button
stop_button = tk.Button(root, text="Reset", justify="center", command=stop_listening, width=20, height=2)
stop_button.pack(side=tk.RIGHT, padx=10, pady=5)

# Label to show verification status
verified_label = tk.Label(root, text="Not Verified", fg="red", font=("Arial", 15))
verified_label.pack(pady=5)

root.mainloop()