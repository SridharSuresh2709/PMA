from pydub import AudioSegment
import speech_recognition as sr

audio_path = "MaleVoice.wav"
audio = AudioSegment.from_wav(audio_path)
chunk_length_ms = 20000  # 20 seconds
chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

r = sr.Recognizer()
full_text = ""

for i, chunk in enumerate(chunks):
    chunk.export(f"chunk{i}.wav", format="wav")
    with sr.AudioFile(f"chunk{i}.wav") as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
            full_text += text + " "
        except sr.UnknownValueError:
            print(f"Chunk {i} could not be understood")
        except sr.RequestError as e:
            print(f"Chunk {i} failed; {e}")

print("Final Transcript:\n")
print(full_text)
