import streamlit as st
import whisper
import tempfile
import os
import subprocess

st.title("🎤 Whisper Transcriber with Video & Audio Extraction")

def run_ffmpeg_command(cmd):
    """Run ffmpeg command, raise error if failed."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")
    return result

@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

uploaded_file = st.file_uploader("Upload video/audio", type=["mp3", "mp4", "wav", "mkv", "mov"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extract audio path
        audio_path = os.path.join(tmpdir, "extracted_audio.wav")
        # Create silent video path
        silent_video_path = os.path.join(tmpdir, "silent_video" + os.path.splitext(uploaded_file.name)[1])

        try:
            # Extract audio from video/audio file (if audio exists)
            cmd_extract_audio = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-vn",  # no video
                "-acodec", "pcm_s16le",
                "-ar", "16000",  # sample rate for Whisper
                "-ac", "1",  # mono audio
                audio_path,
            ]
            run_ffmpeg_command(cmd_extract_audio)

            # Create silent video (video without audio)
            cmd_silent_video = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-an",  # no audio
                silent_video_path,
            ]
            run_ffmpeg_command(cmd_silent_video)

            # Load Whisper model and transcribe
            model = load_model()
            result = model.transcribe(audio_path)
            transcript = result["text"]

            st.success("✅ Transcription done!")
            st.write("### Transcript:")
            st.write(transcript)

            # Display original video if video file
            if uploaded_file.type.startswith("video/"):
                st.video(input_path, format=uploaded_file.type)

                st.download_button(
                    "Download Silent Video (No Audio)",
                    data=open(silent_video_path, "rb").read(),
                    file_name="silent_" + uploaded_file.name,
                    mime=uploaded_file.type,
                )

            # Provide audio download button
            st.audio(audio_path, format="audio/wav")
            st.download_button(
                "Download Extracted Audio",
                data=open(audio_path, "rb").read(),
                file_name="extracted_audio.wav",
                mime="audio/wav",
            )

            # Provide transcript download
            st.download_button(
                "Download Transcript",
                transcript,
                "transcript.txt",
                mime="text/plain",
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")
