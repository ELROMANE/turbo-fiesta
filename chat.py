#!/usr/bin/env python3
"""
Cerebras Chat Demo with Text-to-Speech

A simple chatbot that uses the Cerebras API for responses and can speak them using
text-to-speech. Supports both single-message and interactive REPL modes.

Features:
- Text-to-Speech support (--speak)
- Multi-turn conversation (--repl)
- Conversation history saved to JSON
- Content filtering for speech
- Environment variable or .env file for API key
"""
import argparse
import json
import os
import platform
from pathlib import Path
from typing import List, Dict, Optional

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv


def get_api_key() -> Optional[str]:
    """Get API key from environment or .env file."""
    # Try to load from .env file first
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
    return os.environ.get("CEREBRAS_API_KEY")


def save_history(user_msg: str, assistant_msg: str):
    """Save conversation exchange to JSON history file."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    history_file = data_dir / "conversation_history.json"
    try:
        if history_file.exists():
            history = json.loads(history_file.read_text(encoding="utf-8"))
        else:
            history = []
    except Exception:
        history = []
    history.append({"user": user_msg, "assistant": assistant_msg})
    history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_speaker():
    """Initialize text-to-speech engine with platform-specific settings."""
    try:
        import pyttsx3

        # choose driver based on platform
        system = platform.system()
        print(f"Initializing TTS for {system}...")

        def _filter_for_speech(text: str) -> str:
            """Extract the most relevant parts for speech output."""
            # Split into sentences (basic)
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            # Filter out meta-text and common prefixes
            filtered = []
            skip_patterns = [
                "I'm an AI",
                "I'm just a",
                "I don't have",
                "I cannot",
                "As an AI",
                "let me",
                "please note",
                "note that",
            ]
            
            for sentence in sentences:
                # Skip sentences with meta-text
                if any(pattern.lower() in sentence.lower() for pattern in skip_patterns):
                    continue
                # Skip very short or very long sentences
                if len(sentence) < 5 or len(sentence) > 200:
                    continue
                filtered.append(sentence)
            
            # Combine the most relevant sentences
            if filtered:
                result = ". ".join(filtered[:3]) + "."  # Limit to first 3 relevant sentences
                return result
            return text  # Fallback to original if filtering produced nothing

        def _speak(text: str):
            try:
                if system == "Windows":
                    engine = pyttsx3.init('sapi5')
                elif system == "Darwin":
                    engine = pyttsx3.init('nsss')
                else:
                    engine = pyttsx3.init()
                
                # Get available voices
                voices = engine.getProperty('voices')
                
                # Try to use Zira's voice on Windows, otherwise first available
                voice = next((v for v in voices if 'zira' in v.name.lower()), voices[0])
                engine.setProperty('voice', voice.id)
                
                # Set a slower rate for clarity
                engine.setProperty('rate', 150)  # Default is usually 200
                
                # Filter and speak only the most relevant parts
                filtered_text = _filter_for_speech(text)
                print("\nSpeaking filtered content:")
                print(filtered_text)
                engine.say(filtered_text)
                engine.runAndWait()
            except Exception as e:
                print(f"\nSpeech failed: {e}")

        return _speak
    except Exception as e:
        print(f"\nCould not initialize TTS: {e}")
        return None


def run_once(message: str, speaker=None) -> int:
    """Send a single message and get one response."""
    api_key = get_api_key()
    if not api_key:
        print("Error: CEREBRAS_API_KEY not set. Either:")
        print("1. Set it in your shell:")
        print(r'  $env:CEREBRAS_API_KEY="your-key-here"  # Windows PowerShell')
        print(r'  export CEREBRAS_API_KEY="your-key-here"  # Linux/macOS')
        print("2. Or create a .env file with:")
        print('  CEREBRAS_API_KEY="your-key-here"')
        return 1

    client = Cerebras(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": message}],
            model="llama-4-scout-17b-16e-instruct",
        )
    except Exception as e:
        print("Error calling Cerebras API:", e)
        return 2

    try:
        assistant_reply = resp.choices[0].message.content
    except Exception:
        assistant_reply = str(resp)

    print("\nAssistant:\n", assistant_reply)
    
    if speaker:
        speaker(assistant_reply)
    
    save_history(message, assistant_reply)
    print(f"\nSaved conversation to: {Path(__file__).parent / 'data' / 'conversation_history.json'}")
    return 0


def run_repl(speaker=None) -> int:
    """Run an interactive chat session that preserves context."""
    api_key = get_api_key()
    if not api_key:
        print("Error: CEREBRAS_API_KEY not set. Either:")
        print("1. Set it in your shell:")
        print(r'  $env:CEREBRAS_API_KEY="your-key-here"  # Windows PowerShell')
        print(r'  export CEREBRAS_API_KEY="your-key-here"  # Linux/macOS')
        print("2. Or create a .env file with:")
        print('  CEREBRAS_API_KEY="your-key-here"')
        return 1

    client = Cerebras(api_key=api_key)
    messages: List[Dict[str, str]] = []
    
    print("\nEntering chat REPL. Commands:")
    print("- Type 'exit' or 'quit' to leave")
    print("- Type '/clear' to reset conversation context")
    print("- Just type your message for everything else\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return 0

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            return 0
        if user_input == "/clear":
            messages = []
            print("Context cleared.")
            continue

        # Add user message and call API
        messages.append({"role": "user", "content": user_input})
        try:
            resp = client.chat.completions.create(
                messages=messages,
                model="llama-4-scout-17b-16e-instruct",
            )
        except Exception as e:
            print("Error calling Cerebras API:", e)
            messages.pop()  # Remove failed message
            continue

        try:
            assistant_reply = resp.choices[0].message.content
        except Exception:
            assistant_reply = str(resp)

        # Add assistant reply to context
        messages.append({"role": "assistant", "content": assistant_reply})
        print("\nAssistant:\n", assistant_reply)
        
        if speaker:
            speaker(assistant_reply)
        
        save_history(user_input, assistant_reply)


def main():
    parser = argparse.ArgumentParser(
        description="Cerebras Chat Demo with optional text-to-speech"
    )
    parser.add_argument(
        "--message", "-m",
        help="Send a single message (non-interactive mode)"
    )
    parser.add_argument(
        "--speak", "-s",
        action="store_true",
        help="Enable text-to-speech for assistant replies"
    )
    parser.add_argument(
        "--repl", "-r",
        action="store_true",
        help="Start an interactive chat session (REPL mode)"
    )
    args = parser.parse_args()

    # Initialize TTS if requested
    speaker = _get_speaker() if args.speak else None

    # Choose mode based on args
    if args.repl:
        return run_repl(speaker=speaker)
    elif args.message:
        return run_once(args.message, speaker=speaker)
    else:
        # No mode specified - ask for one message
        try:
            message = input("Enter a message: ")
            return run_once(message, speaker=speaker)
        except KeyboardInterrupt:
            print("\nCancelled")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())