# STEP 1: Install packages (uncomment if needed)
#!pip install transformers gradio torch accelerate gTTS -q
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

#creating the chat function for prompting
def chat_with_assistant(user_input, history): #function that takes the user's msg and history and returns the response
    context = "You are JARVIS, an intelligent and helpful AI assistant. Be conversational and friendly.\n\n" #giving it a personality

    if history:
        for msg in history[-6:]: #adds each past conversation to the context
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                context += f"Human: {content}\n"
            elif role == "assistant":
                context += f"JARVIS: {content}\n" 


    context+= f"Human: {user_input}\nJARVIS:" #adds current input to the context

    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        **inputs,
        max_length = 256, #Longerresponses
        num_beams=4, #Better quality
        temperature=0.7, #more creative
        do_sample=True,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

#creating gradio user interface
def gradio_chat_with_voice(message, history):
    response = chat_with_assistant(message, history)

    try:
        tts = gTTS(text=response, lang="en", slow=False)
        audio_file = "response.mp3"
        tts.save(audio_file)
        return response, audio_file
    except Exception as e:
        print(f"TTS Error: {e}")
        return response, None
    
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
                # 🤖 Your Personal JARVIS Assistant
                ### Built in Google Colab with AI superpowers!
                Ask me anything - I'll respond with text and voice.
                """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=250)

            with gr.Row():
                msg = gr.Textbox(
                    label= "Your message",
                    placeholder="Ask me anything...",
                    scale = 4
                )
                submit = gr.Button("Send 🚀", scale=1, variant="primary")

            clear = gr.Button("Clear Chat 🗑️")

    with gr.Column(scale=1):
        audio_output = gr.Audio(label="🔊 Voice Response", autoplay=True)

        gr.Markdown("### Quick Examples:")
        example_btns = [
            gr.Button("👋 Introduce yourself"),
            gr.Button("😂 Tell me a joke"),
            gr.Button("🧠 Explain AI simply"),
            gr.Button("💡 Give me a fun fact"),
        ]


        #handling chat submissions
        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history, None
            
            bot_response, audio_file = gradio_chat_with_voice(message, chat_history)
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_response})
            return "", chat_history, audio_file
        
    #Wire up events
    msg.submit(respond, [msg, chatbot], [msg, chatbot, audio_output])
    submit.click(respond, [msg, chatbot], [msg, chatbot, audio_output])
    clear.click(lambda: [], None, chatbot, queue=False)
    example_btns[0].click(lambda: "Introduce yourself", None, msg)
    example_btns[1].click(lambda: "Tell me a joke", None, msg)
    example_btns[2].click(lambda: "Explain artificial intelligence in simple terms", None, msg)
    example_btns[3].click(lambda: "Give me an interesting fun fact", None, msg)

print("\n🚀 Launching your AI assistant...")
demo.launch(share=True, debug=True)
