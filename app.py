import streamlit as st
import whisper
import tempfile
import os
import subprocess

st.title("🎤 Whisper Transcriber")

# Check ffmpeg
try:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    st.success("✅ ffmpeg found: " + result.stdout.splitlines()[0])
except Exception as e:
    st.error(f"❌ ffmpeg NOT found: {e}")

uploaded_file = st.file_uploader("Upload audio/video", type=["mp3", "mp4", "wav", "mkv", "mov"])

@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

if uploaded_file:
    st.info("Transcribing...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        model = load_model()
        result = model.transcribe(tmp_path)
        transcript = result["text"]

        st.success("✅ Done!")
        st.write(transcript)

        st.download_button("Download", transcript, "transcript.txt")
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
    finally:
        os.remove(tmp_path)
