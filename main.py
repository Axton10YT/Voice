import os
import asyncio
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import openai
import aiofiles

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Transcribe audio using GPT-4o Transcribe
def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="gpt-4o",
            file=audio_file
        )
    return transcript.text

# Generate text using GPT-5
async def generate_response(prompt):
    response = openai.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a helpful AI voice assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Generate TTS audio using GPT-4o TTS
async def synthesize_speech(text):
    speech_response = openai.audio.speech.create(
        model="tts-1-hd",  # or gpt-4o-mini-tts
        voice="nova",
        input=text,
        response_format="mp3"
    )
    return speech_response.content

@app.post("/talk")
async def talk(audio: UploadFile = File(...)):
    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        audio_path = tmp.name
        async with aiofiles.open(tmp.name, 'wb') as out_file:
            content = await audio.read()
            await out_file.write(content)

    # Transcribe
    transcribed_text = transcribe_audio(audio_path)

    # Generate response
    reply = await generate_response(transcribed_text)

    # Synthesize reply
    mp3_audio = await synthesize_speech(reply)

    # Return streamable MP3
    return StreamingResponse(iter([mp3_audio]), media_type="audio/mpeg")

# Optional: UI to record from browser mic coming next
