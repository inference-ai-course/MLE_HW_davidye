from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import whisper
from transformers import pipeline
import pyttsx3


app = FastAPI()
asr_model = whisper.load_model("tiny")
llm = pipeline("text-generation", model="unsloth/Llama-3.2-1B-Instruct", device_map="auto", torch_dtype="float16")
conversation_history = []
tts_engine = pyttsx3.init()


def transcribe_audio(audio_bytes):
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe("temp.wav")
    return result["text"]

def generate_response(user_text):
    conversation_history.append({"role": "user", "text": user_text})
    prompt = ""
    for turn in conversation_history[-5:]:
        prompt += f"{turn['role']}: {turn['text']}\n"
    outputs = llm(prompt, max_new_tokens=100)
    
    full_response = outputs[0]["generated_text"]
    bot_response = full_response[len(prompt):].strip()

    conversation_history.append({"role": "assistant", "text": bot_response})
    return bot_response

def synthesize_speech(text, filename="response.wav"):
    tts_engine.save_to_file(text, filename)
    tts_engine.runAndWait()
    return filename


@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    
    user_text = transcribe_audio(audio_bytes)
    print(f"user: {user_text}")
    bot_text = generate_response(user_text)
    print(f"bot: {bot_text}")
    audio_path = synthesize_speech(bot_text)
    
    return FileResponse(audio_path, media_type="audio/wav")
