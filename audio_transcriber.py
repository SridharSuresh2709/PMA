from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import google.generativeai as genai
import os
import base64
import io
import wave
import struct
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)


def create_wav_file(pcm_data: bytes, sample_rate: int = 24000, num_channels: int = 1) -> bytes:
    """
    Converts raw PCM audio data into a WAV file.
    
    Args:
        pcm_data: Raw PCM audio bytes
        sample_rate: Sample rate in Hz (default: 24000)
        num_channels: Number of audio channels (default: 1 for mono)
    
    Returns:
        WAV file as bytes
    """
    # Convert bytes to int16 array
    pcm_int16 = struct.unpack(f'{len(pcm_data)//2}h', pcm_data)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    
    wav_buffer.seek(0)
    return wav_buffer.getvalue()


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    from flask import Flask, request, jsonify, send_file
    from flask_cors import CORS
    import google.generativeai as genai
    import os
    import base64
    import io
    import wave
    import struct
    from dotenv import load_dotenv

    load_dotenv()

    app = Flask(__name__)
    CORS(app)

    # Configure Gemini API
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    genai.configure(api_key=GEMINI_API_KEY)


    def create_wav_file(pcm_data: bytes, sample_rate: int = 24000, num_channels: int = 1) -> bytes:
        """
        Converts raw PCM audio data into a WAV file.
    
        Args:
            pcm_data: Raw PCM audio bytes
            sample_rate: Sample rate in Hz (default: 24000)
            num_channels: Number of audio channels (default: 1 for mono)
    
        Returns:
            WAV file as bytes
        """
        # Convert bytes to int16 array
        pcm_int16 = struct.unpack(f'{len(pcm_data)//2}h', pcm_data)
    
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
    
        wav_buffer.seek(0)
        return wav_buffer.getvalue()


    @app.route('/api/transcribe', methods=['POST'])
    def transcribe_audio():
        """
        Endpoint to transcribe audio file to text.
    
        Expects:
            - audio_data: base64 encoded audio data
            - mime_type: MIME type of the audio file
    
        Returns:
            JSON with transcript
        """
        try:
            data = request.get_json()
        
            if not data or 'audio_data' not in data or 'mime_type' not in data:
                return jsonify({'error': 'Missing audio_data or mime_type'}), 400
        
            audio_data = data['audio_data']
            mime_type = data['mime_type']
        
            # Prepare audio part for Gemini
            audio_part = {
                "mime_type": mime_type,
                "data": audio_data
            }
        
            # Generate transcript using Gemini
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content([
                audio_part,
                "Please provide a clean, accurate transcription of the following audio."
            ])
        
            transcript = response.text
        
            return jsonify({
                'transcript': transcript,
                'success': True
            })
    
        except Exception as e:
            return jsonify({
                'error': str(e),
                'success': False
            }), 500



    @app.route('/api/synthesize', methods=['POST'])
    def synthesize_speech():
        """
        Endpoint to synthesize speech from text.
    
        Expects:
            - text: Text to convert to speech
    
        Returns:
            JSON with base64 encoded audio data
        """
        try:
            data = request.get_json()
        
            if not data or 'text' not in data:
                return jsonify({'error': 'Missing text parameter'}), 400
        
            text = data['text']
        
            # Generate speech using Gemini TTS
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
            response = model.generate_content(
                f"Please say the following: {text}",
                generation_config=genai.GenerationConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai.SpeechConfig(
                        voice_config=genai.VoiceConfig(
                            prebuilt_voice_config=genai.PrebuiltVoiceConfig(
                                voice_name="Kore"
                            )
                        )
                    )
                )
            )
        
            # Extract audio data
            if response.candidates and len(response.candidates) > 0:
                audio_data = response.candidates[0].content.parts[0].inline_data.data
            
                return jsonify({
                    'audio_data': audio_data,
                    'success': True
                })
            else:
                return jsonify({
                    'error': 'No audio data generated',
                    'success': False
                }), 500
    
        except Exception as e:
            return jsonify({
                'error': str(e),
                'success': False
            }), 500



    @app.route('/api/process', methods=['POST'])
    def process_audio():
        """
        Combined endpoint: transcribe audio and synthesize speech.
    
        Expects:
            - audio_data: base64 encoded audio data
            - mime_type: MIME type of the audio file
    
        Returns:
            JSON with transcript and synthesized audio
        """
        try:
            data = request.get_json()
        
            if not data or 'audio_data' not in data or 'mime_type' not in data:
                return jsonify({'error': 'Missing audio_data or mime_type'}), 400
        
            audio_data = data['audio_data']
            mime_type = data['mime_type']
        
            # Step 1: Transcribe audio
            audio_part = {
                "mime_type": mime_type,
                "data": audio_data
            }
        
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            transcript_response = model.generate_content([
                audio_part,
                "Please provide a clean, accurate transcription of the following audio."
            ])
        
            transcript = transcript_response.text
        
            # Step 2: Synthesize speech from transcript
            speech_response = model.generate_content(
                f"Please say the following: {transcript}",
                generation_config=genai.GenerationConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai.SpeechConfig(
                        voice_config=genai.VoiceConfig(
                            prebuilt_voice_config=genai.PrebuiltVoiceConfig(
                                voice_name="Kore"
                            )
                        )
                    )
                )
            )
        
            # Extract synthesized audio
            synthesized_audio = None
            if speech_response.candidates and len(speech_response.candidates) > 0:
                synthesized_audio = speech_response.candidates[0].content.parts[0].inline_data.data
        
            return jsonify({
                'transcript': transcript,
                'audio_data': synthesized_audio,
                'success': True
            })
    
        except Exception as e:
            return jsonify({
                'error': str(e),
                'success': False
            }), 500



    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({'status': 'healthy'})


    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)