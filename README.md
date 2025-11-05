# Personalized Memory Assistant (PMA)

A Personalized Memory Assistant powered by AI - Setup notes for Windows (PowerShell)

These quick steps create a virtual environment, install dependencies, and cover Windows-specific notes (PyAudio and ffmpeg).

1) Create & activate a venv

```powershell
python -m venv .venv
# Activate venv (PowerShell)
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
```

2) Install Python packages from the full requirements file

```powershell
pip install -r .\requirements_full.txt
```

3) Install PyTorch (CPU-only) — required by sentence-transformers

```powershell
# CPU-only PyTorch (recommended if you don't have CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

If you have an NVIDIA GPU with CUDA, follow the official instructions at https://pytorch.org to pick the wheel matching your CUDA version.

4) Install PyAudio on Windows (recommended via pipwin)

```powershell
pip install pipwin
pipwin install pyaudio
```

If `pipwin` fails, download a prebuilt PyAudio wheel matching your Python version from a trusted wheel archive and install with `pip install <wheel-file>`.

5) ffmpeg (required by pydub)

- Install with Chocolatey (if you use it):

```powershell
choco install ffmpeg -y
```

- Or download a static ffmpeg build from https://ffmpeg.org/download.html and add the `bin` folder to your PATH.

Verification

```powershell
# quick import test
python -c "import chromadb, sentence_transformers, speech_recognition, pyttsx3, google.generativeai; print('OK')"

# check ffmpeg is on PATH
ffmpeg -version
```

Environment variables

- The code expects an API key in `GEMINI_API_KEY` or `GOOGLE_API_KEY` (the project already supports loading a local `.env`).
- For PowerShell (current session):

```powershell
$env:GEMINI_API_KEY = "your_api_key_here"
```

- Persistently (user-level):

```powershell
setx GEMINI_API_KEY "your_api_key_here"
```

Security

- Do not commit real secrets. `.env` is added to `.gitignore` by the project. If you've committed a key previously, rotate it now.
