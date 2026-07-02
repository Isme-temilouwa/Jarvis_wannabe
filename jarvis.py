import os
import json
from anthropic import Anthropic
from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()
import sounddevice as sd
from scipy.io.wavfile import write
from groq import Groq
import numpy as np
import pyttsx3
from datetime import datetime, timedelta
import threading
import time


lock = threading.Lock()

with open('memory.json', 'r') as f:
    facts = json.load(f)

try:
    with open('history.json','r') as f:
        convo = json.load(f)
except (FileNotFoundError,json.JSONDecodeError):
    convo = []
    
    
try:
    with open('remind.json','r') as f:
        reminders = json.load(f)
    for reminder in reminders:
        reminder['due'] = datetime.fromisoformat(reminder['due'])
except (FileNotFoundError, json.JSONDecodeError):
    reminders = []
        
    
engine = pyttsx3.init()
client = Anthropic(api_key = os.environ.get("DEEPSEEK_API_KEY"), base_url = "https://api.deepseek.com/anthropic")
tavily_client = TavilyClient(api_key=os.environ.get('TAVILY_API_KEY'), api_base_url='https://api.tavily.com')
groq_client = Groq(api_key=os.environ.get('GROK_API_KEY'))



def check_logic():
    global reminders
    while True:
        still_pending = []
        time.sleep(30)
        with lock:
            for reminder in reminders:
                if datetime.now() >= reminder['due']:
                    speak('Reminder: ' + reminder['content'])
                else:
                    still_pending.append(reminder)
            reminders = still_pending
thread = threading.Thread(target=check_logic)
thread.start()


def check_reminders():
    with lock:
        if not reminders:
            return "You have no pending reminders."
        output = ''
        for reminder in reminders:
            output += 'Reminder: ' + reminder['content'] + ', due ' + str(reminder['due']) + '\n'
        return output
        

def listen():
    fs = 44100
    recording = []
    def callback(indata, frames, time, status):
        recording.append(indata.copy())
    print('recording...press enter to stop')
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        input()
    audio = np.concatenate(recording)
    write('record.wav', fs, audio)
    with open('record.wav','rb') as f:
        transcription = groq_client.audio.transcriptions.create(model="whisper-large-v3-turbo", file=f, language= 'en')
    return transcription.text
        
        
def set_reminder(content, minutes):
    due = datetime.now() + timedelta(minutes=minutes)
    reminder = {'content':content, 'due':due}
    with lock:
        reminders.append(reminder)
    return f"I will remind you about  {content}"




def calculator ( num1, num2, operator):
    num1 = float(num1)
    num2 = float(num2)
    if operator == "+":
        return num1 + num2
    elif operator == "-":
        return num1 - num2
    elif operator == "*":
        return num1 * num2
    elif operator == "/":
        if num2 == 0:
            return ("Can't be divisible by 0")
        else:
            return num1/num2
    elif operator == "^":
        return num1 ** num2
    else:
        return ("Unrecognized number or operator" )
    
def search (query):
    search_response = tavily_client.search(query=query,search_depth='basic',max_results=3,chunks_per_source=4)
    answers = search_response['results']
    if not answers:
        return "No results found"
    output = ''
    for answer in answers:
        output += "Title: " + answer['title'] + "\n"
        output += "URL: " + answer['url'] + "\n"
        output += "Content: " + answer['content'] + "\n\n"

    return output

def writer(key,value):
    existing_data = {
    "name": "Temiloluwa",
}
    if os.path.exists('memory.json'):
        with open('memory.json', 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                pass
    existing_data[key] = value
    with open('memory.json','w') as f:
        json.dump(existing_data,f)
    return 'I learnt something new about you today, i do well to remember that'




def speak(text):
    engine.say(text)
    engine.runAndWait()
    
    
tools = [
    {
        "name" : "Calculator",
        "description" : "This is an arithmetic tool that does calculations, it takes numbers and an operator and outputs the answer.",
        "input_schema" : {
            "type": "object",
            "properties": {
                "num1" : {"type": "number"},
                "num2" : {"type": "number"},
                "operator": {"type": "string"},
            },
            "required" : ['num1', "num2", "operator"],
        },
    },
    {
        "name": 'Searcher',
        'description': "This is a tool that helps to search the web for whatever the users asks and you dont know.",
        'input_schema': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string'},
            },
            'required': ['query']
            
        },
    },
    
    {
        "name": 'Remember',
        'description': "Use this tool to save any fact or preference about the user to long-term memory, so you remember it in future conversations. Provide a short descriptive 'key' and the fact as the 'value'. For example: key='favorite_sport', value='basketball', or key='partner_name', value='Sylvia', or key='dietary_preference', value='vegetarian'. Call it whenever the user shares something worth remembering or asks you to remember something. provide a short key and the fact as the value, e.g. key='favorite_sport' value='basketball', or key='partner_name' value='Sylvia gives it the shape. This is very important.",
        'input_schema': {
            'type': 'object',
            'properties': {
                'key': {'type' : 'string', 'description': 'A short label for the fact, e.g. "favorite_sport"'},
                'value' : {'type' : 'string', 'description': 'The fact itself, e.g. "basketball"' }
            },
            'required': ['key','value']
        
        },
    },
    
    {
        'name': 'Remind_Me',
        'description': (
    "Creates a reminder that will trigger after a specified number of minutes. "
    "Use this tool whenever the user asks to be reminded later. "
    "The 'content' parameter is what to remind the user about. "
    "The 'minutes' parameter is the number of minutes from now. "
    "After the tool succeeds, tell the user that the reminder has been set."
),
        'input_schema': {
            'type': 'object',
            'properties': {
                'content': {'type': 'string'},
                'minutes': {'type': 'number'},
            },
            'required' : ['content','minutes'],
        },
    },
    
    
   {
    'name': 'Check_Reminders',
    'description': "Use this tool whenever the user asks what reminders they have, what's pending, or wants to see their current reminders. It reads the live list of active reminders and returns them. Call this instead of guessing from the conversation, it reflects the actual pending reminders in the system.",
    'input_schema': {
        'type': 'object',
        'properties': {},
    },
},
    
]

system_prompt = "You are a helpful assistant and personal assistant to the user. i built you. You have a remind_me tool make sure you use it whenever i ask to be reminded.  Here is what you know about them:\n"
for key in facts:
    system_prompt += key + ": " + facts[key] + "\n"
    
    


while True:
    
    choice = input('What do you want to use to speak to me: Voice(v) or Text(t)').lower()
    if choice == 't':
        user_input = input("What is your prompt: ")
    elif choice == 'v':
        user_input = listen()
    else:
        user_input = input("What is your prompt: ")
    convo.append({'role':'user', 'content':user_input})
    if user_input.lower().strip().strip('.!?,') == 'quit':
        break
    try:
        response = client.messages.create(model="deepseek-v4-flash", max_tokens=1024, system = system_prompt , tools=tools,  messages=convo)
    except Exception as e:
        print(e)
    while response.stop_reason == 'tool_use': 
        for block in response.content:
            if block.type == 'tool_use':
                tool_block = block
        if tool_block.name == 'Calculator':
            try:
                answer = calculator(tool_block.input.get('num1'),tool_block.input.get('num2'), tool_block.input.get('operator'))
            except Exception as e:
                answer = "The calculation couldn't be calculated"
                print(e)
        elif tool_block.name == "Searcher":
            try:
                answer = search(tool_block.input.get('query'))
            except Exception as e:
                answer = "i cannot get you an answer right now because something went wrong"
                print(e)
        elif tool_block.name == 'Remember':
            try:
                answer = writer(tool_block.input.get('key'),tool_block.input.get('value') )
            except Exception as e:
                answer = 'Could not save that to memory.'
                print(e)
        elif tool_block.name == 'Remind_Me':
            try:
                answer = set_reminder(tool_block.input.get('content'), tool_block.input.get('minutes'))
                
            except Exception as e:
                answer = 'Could not set your reminder'
        elif tool_block.name == 'Check_Reminders':
            try:
                answer = check_reminders()
            except Exception as e:
                answer = 'Cant check your pending reminders'
        else:
            answer = "Unknown tool requested: " + tool_block.name
        convo.append({'role':'assistant', 'content':response.content})
        convo.append({'role':'user', 'content': [{
            'type': 'tool_result',
            'tool_use_id': tool_block.id,
            'content': str(answer)    
            }]})
        response = client.messages.create(model="deepseek-v4-flash", max_tokens=1024,tools=tools,system = system_prompt , messages=convo)
    for block in response.content:
        if block.type == 'text':
            replies = block
    try:
        reply = replies.text
    except Exception as e:
        reply = "Couldn't generate reply, wait a while and ask again."
        print('DEBUG:',e)
    print(reply)
    if choice == 'v':
        speak(reply)
    convo.append({'role':'assistant', 'content': reply})
    
    save_reminders = []
    for reminder in reminders:
        remind = {'content':reminder['content'], 'due' : reminder['due'].isoformat()}
        save_reminders.append(remind)
    with open('remind.json','w') as f:
        json.dump(save_reminders,f)
        
    clean_convo = []
    
    for entry in convo:
        if isinstance(entry['content'],str):
            clean_convo.append(entry)
        else:
            clean_convo.append({'role': entry['role'],'content': '[tool was used]'})
    with open('history.json', 'w') as f:
        json.dump(clean_convo,f)    

