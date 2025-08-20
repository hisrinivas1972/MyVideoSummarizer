import os
os.system("apt-get update && apt-get install -y ffmpeg")

import os
import subprocess
import streamlit as st
import whisper
import tempfile

# — Force-install ffmpeg at runtime to avoid deployment issues
os.system("apt-get update && apt-get install -y ffmpeg")

st.title("🎤 Whisper Transcriber")

# Confirm ffmpeg is indeed installed
try:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    st.success("✅ ffmpeg found: " + result.stdout.splitlines()[0])
except Exception as e:
    st.error(f"❌ ffmpeg NOT found: {e}")

# File uploader
uploaded_file = st.file_uploader("Upload audio/video file", type=["mp3", "mp4", "mkv", "wav", "mov"])

@st.cache_resource
def load_model():
    return whisper.load_model("tiny")  # Or "base" / "small" for more accuracy

if uploaded_file:
    st.info("Transcribing...")

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    try:
        model = load_model()
        result = model.transcribe(temp_path)
        transcript = result["text"]

        st.success("✅ Transcription done!")
        st.subheader("📝 Transcript")
        st.write(transcript)
        st.download_button("Download Transcript", transcript, "transcript.txt")
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
    finally:
        os.remove(temp_path)
