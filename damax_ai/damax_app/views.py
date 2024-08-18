from django.shortcuts import render
from django.http import JsonResponse
from .forms import VideoUploadForm
import whisper
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from moviepy.editor import VideoFileClip
from django.core.files.storage import default_storage
import os

# Global variables
model = whisper.load_model("base")

template = """Use the following pieces of context to answer the question at the end, the context could represent audio, video, meeting or audiobook,
the context is annotated with timestamps, the answer should only mention the time of user asked about it.
If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

def create_retrieval_qa_chain():
    loader = TextLoader("media/output.txt", encoding='UTF-8')
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(splits, embedding_function)
    retriever = db.as_retriever()

    ollama = Ollama(base_url='http://localhost:11434', model="llama3")
    qachain = RetrievalQA.from_chain_type(llm=ollama, retriever=retriever, chain_type="stuff")
    return qachain

def transcribe_audio(audio_path):
    transcript = model.transcribe(audio=audio_path, word_timestamps=True)
    
    # Add timestamps to each segment
    for segment in transcript['segments']:
        start_minutes, start_seconds = divmod(segment["start"], 60)
        timestamp = f"{int(start_minutes):02d}:{int(start_seconds):02d}"
        segment["timestamp"] = timestamp

    transcription_text = " ".join(segment["timestamp"] + ": " + segment["text"] for segment in transcript['segments'])

    with open("media/output.txt", "w") as text_file:
        text_file.write(transcription_text)

def process_video(file_path):
    audio_path = "media/extracted_audio.wav"
    try:
        video = VideoFileClip(file_path)
        video.audio.write_audiofile(audio_path)
        transcribe_audio(audio_path)
    except Exception as e:
        return str(e)
    return "Success"

def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.cleaned_data['video']
            video_path = default_storage.save(video.name, video)
            processing_result = process_video("media/"+video_path)
            if processing_result == "Success":
                return JsonResponse({'status': 'success', 'redirect_url': '/video/ask/'})
            else:
                return JsonResponse({'status': 'error', 'message': processing_result})
    else:
        form = VideoUploadForm()
    return render(request, 'damax_app/upload.html', {'form': form})

def ask_question(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        if question:
            qachain = create_retrieval_qa_chain()
            answer = gen_answer(qachain, question)
            return JsonResponse({'status': 'success', 'answer': answer})
        else:
            return JsonResponse({'status': 'error', 'message': 'Please enter a question.'})
    return render(request, 'damax_app/ask.html')

def gen_answer(qachain, question):
    res = qachain.invoke({"query": question})
    return res['result']
