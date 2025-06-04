import whisper

def transcribe_with_whisper(audio_path):
    model = whisper.load_model("base")  # Use "small", "medium", or "large" for better accuracy
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]

def main():
    aac_path = "a.aac"  # Replace with your file
    transcription = transcribe_with_whisper(aac_path)
    print("Transcription:\n", transcription)

if __name__ == "__main__":
    main()