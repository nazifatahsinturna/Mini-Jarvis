from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import gradio as gr #for user interface
import torch
from gtts import gTTS
import os

print("Loading your AI Assistant (This might take a minute)")

#loading the AI Model
model_name ="google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("Assitant loaded! Ready to chat.")

