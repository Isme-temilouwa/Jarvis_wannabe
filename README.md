A voice-enabled personal AI assistant I'm building in Python.
It runs on an agent loop and has a set of tools: web search, memory, voice input and output, and reminders. You can talk to it or type to it, and it talks back.
I'm building it for two reasons. One, to make my own life a bit easier. Two, to sharpen my coding on the way to becoming an AI engineer. It's the project I use to turn understanding into actually being able to build things from scratch.

Tech stack
Python
DeepSeek for the language model
Tavily for web search
Groq Whisper for speech-to-text
pyttsx3 for text-to-speech
threading for the background reminders that fire in real time

Status
Work in progress. I'm building it in public, one piece at a time.
Running it

Clone the repo.
Install the dependencies: pip install -r requirements.txt
Create a .env file with your API keys:

DEEPSEEK_API_KEY=your_key
TAVILY_API_KEY=your_key
GROK_API_KEY=your_key

Run it: python jarvis.py
