import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import whisper
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import os
import platform
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from PIL import Image, ImageTk

# Global variables
audio_path = "audio/extracted_audio.wav"
transcription_path = "text/output.txt"

# Load the Whisper model
model = whisper.load_model("base")

# Define the prompt template
template = """Use the following pieces of context to answer the question at the end, the context could represent audio, video, meeting or audiobook,
the context is annotated with timestamps, the answer should only mention the time of user asked about it.
If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

def create_retrieval_qa_chain():
    loader = TextLoader(transcription_path, encoding='UTF-8')
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(splits, embedding_function)
    retriever = db.as_retriever()

    ollama = Ollama(base_url='http://localhost:11434', model="llama3")
    qachain = RetrievalQA.from_chain_type(llm=ollama, retriever=retriever, chain_type="stuff")
    return qachain

def transcribe_audio():
    transcript = model.transcribe(audio=audio_path, word_timestamps=True)
    
    # Add timestamps to each segment
    for segment in transcript['segments']:
        start_minutes, start_seconds = divmod(segment["start"], 60)
        timestamp = f"{int(start_minutes):02d}:{int(start_seconds):02d}"
        segment["timestamp"] = timestamp

    transcription_text = " ".join(segment["timestamp"] + ": " + segment["text"] for segment in transcript['segments'])

    with open(transcription_path, "w") as text_file:
        text_file.write(transcription_text)

def gen_answer(qachain, question):
    res = qachain.invoke({"query": question})
    return res['result']

def gen_audio(answer):
    audio_array = generate_audio(answer)
    write_wav("gen_audio/bark_generation.wav", SAMPLE_RATE, audio_array)

def text_to_speech(text):
    def play_audio():
        # Create a gTTS object
        tts = gTTS(text=text, lang='en')

        # Save the audio file
        audio_file = "output.mp3"
        tts.save(audio_file)
        if platform.system() == 'Linux' and 'microsoft' in platform.uname().release.lower():
            # Play the audio file using ffplay
            os.system(f'ffplay -autoexit -nodisp {audio_file}')
        else: 
            # Load and play the audio file
            audio = AudioSegment.from_mp3(audio_file)
            play(audio)
        # Clean up
        os.remove(audio_file)

    threading.Thread(target=play_audio).start()

def process_video(file_path):
    global audio_path
    audio_path = "audio/" + os.path.splitext(os.path.basename(file_path))[0] + "_extracted_audio.wav"
    try:
        video = VideoFileClip(file_path)
        video.audio.write_audiofile(audio_path)
        transcribe_audio()
        messagebox.showinfo("Success", "Processing completed successfully.")
        enable_question_input()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process the file: {e}")

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        status_label.config(text="Processing video...")
        threading.Thread(target=process_video, args=(file_path,)).start()

def ask_question():
    def process_question():
        question = question_entry.get()
        if question:
            answer_text.set("Answer:")
            answer_text.set("Could you give me a moment? I'm thinking..")
            qachain = create_retrieval_qa_chain()
            answer = gen_answer(qachain, question)
            answer_text.set(answer)
            text_to_speech_button.pack(pady=5)  # Show the text-to-speech button when answer is ready
        else:
            messagebox.showerror("Error", "Please enter a question.")

    threading.Thread(target=process_question).start()

def enable_question_input():
    status_label.config(text="Processing complete. You can now ask questions.")
    question_label.pack(pady=5)
    question_entry.pack(pady=5)
    ask_button.pack(pady=10)
    speech_recognition_button.pack(pady=10)
    #answer_label.pack(pady=5)
    answer_box.pack(pady=5)

def speech_recognition():
    def recognize_speech():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Adjusting for ambient noise. Please wait...")
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            speech_recognition_button.config(text="Listening...", state=tk.DISABLED)
            audio = recognizer.listen(source)
            try:
                print("Recognizing...")
                text = recognizer.recognize_google(audio)
                question_entry.delete(0, tk.END)
                question_entry.insert(0, text)
                messagebox.showinfo("Recognition Result", f"You said: {text}")
            except sr.UnknownValueError:
                messagebox.showerror("Error", "Sorry, I could not understand the audio.")
            except sr.RequestError:
                messagebox.showerror("Error", "Sorry, there was an issue with the request.")
            finally:
                speech_recognition_button.config(text="Use Speech Recognition", state=tk.NORMAL)
    
    threading.Thread(target=recognize_speech).start()

# Create the main window
root = tk.Tk()
root.title("Damax")
# Load and set the application icon
icon_image = Image.open("logos/1.png")
icon_photo = ImageTk.PhotoImage(icon_image)
root.iconphoto(True, icon_photo)
# Set the initial size of the window
root.geometry("1000x800") 

# Load and resize the logo image
logo_image = Image.open("logos/1.jpg")
logo_image = logo_image.resize((300, 300), Image.LANCZOS)  # Resize to desired dimensions
logo_photo = ImageTk.PhotoImage(logo_image)
# Create and place widgets with dark theme
logo_label = tk.Label(root, image=logo_photo)
logo_label.pack(pady=10)

# Create and place widgets
file_button = tk.Button(root, text="Select Video File", command=select_file)
file_button.pack(pady=10)

status_label = tk.Label(root, text="")
status_label.pack(pady=10)

question_label = tk.Label(root, text="Enter your question:")
question_entry = tk.Entry(root, width=50)
ask_button = tk.Button(root, text="Ask", command=ask_question)
speech_recognition_button = tk.Button(root, text="Use Speech Recognition", command=speech_recognition)

answer_text = tk.StringVar()
#answer_label = tk.Label(root, text="")
answer_box = tk.Label(root, textvariable=answer_text, wraplength=400)

# Button for text-to-speech
text_to_speech_button = tk.Button(root, text="Read Answer Aloud", command=lambda: text_to_speech(answer_text.get()))
text_to_speech_button.pack_forget()  # Initially hide the button

# Start the GUI event loop
root.mainloop()
