import streamlit as st
import whisper
import tempfile
import os
import subprocess

st.title("🎤 Whisper Transcriber")

def has_audio(file_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return "audio" in result.stdout

uploaded_file = st.file_uploader("Upload audio/video", type=["mp3", "mp4", "wav", "mkv", "mov"])

@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if not has_audio(tmp_path):
        st.error("❌ Uploaded video does not contain any audio track. Please upload a file with audio.")
        os.remove(tmp_path)
    else:
        st.info("Transcribing...")

        try:
            model = load_model()
            result = model.transcribe(tmp_path)
            transcript = result["text"]

            st.success("✅ Done!")
            st.write(transcript)
            st.download_button("Download transcript", transcript, "transcript.txt")
        except Exception as e:
            st.error(f"❌ Transcription error: {e}")
        finally:
            os.remove(tmp_path)
