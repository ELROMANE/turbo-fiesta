<<<<<<< HEAD
# Cerebras Chat Demo

A simple chatbot with text-to-speech support, using the Cerebras API for responses.

## Features

- Text-to-speech support (speaks the assistant's replies)
- Multi-turn conversation with context memory
- Filters out AI meta-text from speech output
- Saves conversation history to JSON
- Cross-platform TTS support (Windows, macOS, Linux)
- API key from environment or .env file

## Setup

1. **Create a virtual environment:**
   ```powershell
   # Windows
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux/macOS
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key:**
   - Copy `.env.example` to `.env`
   - Edit `.env` and add your Cerebras API key
   - Or set it in your shell:
     ```powershell
     # Windows PowerShell
     $env:CEREBRAS_API_KEY="your-key-here"
     
     # Linux/macOS
     export CEREBRAS_API_KEY="your-key-here"
     ```

## Usage

### Single message mode

```bash
# Just chat (no speech)
python chat.py --message "Hello, how are you?"

# Chat with speech
python chat.py --message "Hello, how are you?" --speak
```

### Interactive REPL mode (with context memory)

```bash
# Start REPL without speech
python chat.py --repl

# Start REPL with speech
python chat.py --repl --speak
```

### REPL Commands

- Type `exit` or `quit` to leave
- Type `/clear` to reset conversation context
- Just type your message for everything else

## Files

- `chat.py` - Main script
- `requirements.txt` - Python package dependencies
- `.env` - Your API key (create from .env.example)
- `data/conversation_history.json` - Saved chat history

## Notes

- Text-to-speech uses Windows SAPI5 (Zira's voice) on Windows, macOS voices on Mac, and espeak/festival on Linux
- Speech output is filtered to remove common AI meta-text ("As an AI...", etc.)
- Only the first 3 relevant sentences are spoken for clarity
- Full conversation context is preserved in REPL mode

## Training data generator and math OCR tools

This repository also contains tools to generate synthetic training data for math OCR, validate it, export smaller sample sets with manifests, and convert to TFRecords for training.

Key files added/updated:

- `training_data_generator.py` — extended expression templates (fractions, nested superscripts/subscripts, Greek/math symbols) and a `generate_sample_set()` helper that writes `manifest.csv` and `manifest.json`.
- `dataset_export_tfrecords.py` — optional TFRecord converter (requires TensorFlow). The script will instruct how to install TensorFlow if it's missing.
- `dataset_loader.py` — a tiny manifest-based loader that yields PIL images and labels.
- `fonts/README.txt` — instructions to add a math-capable font named `math_font.ttf` to `./fonts` for reproducible rendering.

Usage notes:

1. (Optional) Add a math-capable font:
   - Place a math TTF/OTF in `fonts/` and rename it to `math_font.ttf`.
   - Recommended fonts: STIX Two Math, Latin Modern Math (check license before bundling).

2. Generate the full dataset (default 100 samples/category):
   - Run: `python training_data_generator.py`

3. Generate the small sample set (50 samples/category) with manifest:
   - From Python REPL or a script:
     ```python
     from training_data_generator import MathTrainingDataGenerator
     gen = MathTrainingDataGenerator()
     gen.generate_sample_set(samples_per_category=50)
     ```

4. Convert to TFRecords (optional, requires TensorFlow):
   - Example:
     `python dataset_export_tfrecords.py --input-dir data/training --output data/train.tfrecord`

5. Load samples with `dataset_loader.SimpleDatasetLoader(manifest_csv='data/training/sample/manifest.csv')`.

If you'd like, I can add tests for the generator/validator, or attempt an automated font download (requires network access and license confirmation).
=======