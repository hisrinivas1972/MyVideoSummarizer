import streamlit as st
import whisper
import tempfile
import os
import subprocess

# Centered title
st.markdown("<h1 style='text-align: center;'>🎤 Whisper Transcriber with Video & Audio Extraction</h1>", unsafe_allow_html=True)

# Copyright reminder below heading, centered and styled subtly
st.markdown(
    "<p style='text-align: center; font-style: italic; color: gray; margin-top: -10px;'>"
    "⚠️ Please follow copyright and usage guidelines before uploading any content."
    "</p>", 
    unsafe_allow_html=True
)
def run_ffmpeg_command(cmd):
    """Run ffmpeg command, raise error if failed."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")
    return result

@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

# File uploader in sidebar
uploaded_file = st.sidebar.file_uploader("Upload video/audio", type=["mp3", "mp4", "wav", "mkv", "mov"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        audio_path = os.path.join(tmpdir, "extracted_audio.wav")
        silent_video_path = os.path.join(tmpdir, "silent_video" + os.path.splitext(uploaded_file.name)[1])

        try:
            # Extract audio (for both video and audio files)
            cmd_extract_audio = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                audio_path,
            ]
            run_ffmpeg_command(cmd_extract_audio)

            # Create silent video (video without audio)
            cmd_silent_video = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-an",
                silent_video_path,
            ]
            run_ffmpeg_command(cmd_silent_video)

            # Transcribe audio with Whisper
            model = load_model()
            result = model.transcribe(audio_path)
            transcript = result["text"]

            st.success("✅ Transcription done!")

            st.write("### Transcript:")
            st.write(transcript)

            # Show original video if uploaded file is a video
            if uploaded_file.type.startswith("video/"):
                st.video(input_path, format=uploaded_file.type)

                st.download_button(
                    "Download Silent Video (No Audio)",
                    data=open(silent_video_path, "rb").read(),
                    file_name="silent_" + uploaded_file.name,
                    mime=uploaded_file.type,
                )

            # Show extracted audio playback + download
            st.audio(audio_path, format="audio/wav")
            st.download_button(
                "Download Extracted Audio",
                data=open(audio_path, "rb").read(),
                file_name="extracted_audio.wav",
                mime="audio/wav",
            )

            # Download transcript
            st.download_button(
                "Download Transcript",
                transcript,
                "transcript.txt",
                mime="text/plain",
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")
