import gradio as gr
import torch
import os
from lmdeploy import pipeline, TurbomindEngineConfig

base_path = './internlm2-1_8b-pet-knowledge-assistant-model/'
os.system('apt install git')
os.system('apt install git-lfs')
os.system(f'git clone https://code.openxlab.org.cn/raytang88/internlm2-1_8b-pet-knowledge-assistant-model.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

backend_config = TurbomindEngineConfig(session_len=8192, model_fomat='hf')

pipe = pipeline(base_path, backend_config=backend_config)

def model(text):
    if text is None:
        return [(text, "请输入你的问题。")]
    else:
        response = pipe((text)).text
    return [(text, response)]

# demo = gr.Interface(fn=model, inputs=[gr.Image(type="pil"), gr.Textbox()], outputs=gr.Chatbot())
demo = gr.Interface(fn=model, inputs=[gr.Textbox()], outputs=gr.Chatbot())
demo.launch()


    