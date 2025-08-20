import os
os.system("apt-get update && apt-get install -y ffmpeg")

import os
import subprocess
import streamlit as st
import whisper
import tempfile

# 1) Force-install ffmpeg at runtime
os.system("apt-get update && apt-get install -y ffmpeg")

# 2) App layout
st.set_page_config(page_title="🎤 Whisper Transcriber", layout="centered")
st.title("🎤 Whisper Video/Audio Transcriber")

# 3) Verify ffmpeg installation
try:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    st.success("✅ ffmpeg found: " + result.stdout.splitlines()[0])
except Exception as e:
    st.error(f"❌ ffmpeg NOT found: {e}")

# 4) File uploader
uploaded_file = st.file_uploader(
    "Upload an audio or video file", 
    type=["mp3", "mp4", "mkv", "wav", "mov"]
)

@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

if uploaded_file:
    st.info("Transcribing... Please wait ⏳")

    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    try:
        model = load_model()
        result = model.transcribe(temp_path)
        transcript = result["text"]

        st.success("✅ Transcription completed!")
        st.subheader("📝 Transcript")
        st.write(transcript)

        st.download_button("📄 Download TXT", transcript, file_name="transcript.txt", mime="text/plain")
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
    finally:
        os.remove(temp_path)
