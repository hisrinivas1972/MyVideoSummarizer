import os

# ✅ Force-install ffmpeg at the start
os.system("apt-get update && apt-get install -y ffmpeg")

import streamlit as st
import whisper
import tempfile
import subprocess

st.set_page_config(page_title="🎤 Whisper Transcriber", layout="centered")
st.title("🎤 Whisper Video/Audio Transcriber")

# ✅ Check if ffmpeg was installed
try:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    st.success("✅ ffmpeg found: " + result.stdout.splitlines()[0])
except Exception as e:
    st.error(f"❌ ffmpeg NOT found: {e}")

# Upload video/audio
uploaded_file = st.file_uploader("Upload a file", type=["mp4", "mp3", "wav", "mkv", "mov"])

@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

if uploaded_file:
    st.info("Transcribing...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    try:
        model = load_model()
        result = model.transcribe(temp_path)
        transcript = result["text"]

        st.success("✅ Done!")
        st.subheader("Transcript")
        st.write(transcript)

        st.download_button("Download as TXT", transcript, file_name="transcript.txt", mime="text/plain")

    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
    finally:
        os.remove(temp_path)

