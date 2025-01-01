import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline

text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6",
                torch_dtype = torch.bfloat16)

def summary (input):
    output = text_summary(input)
    return output[0]['summary_text']

gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text", outputs="text")
demo = gr.Interface(fn=summary,
                    inputs=[gr.Textbox(label="Input the text to summarize")],
                    outputs=[gr.Textbox(label="Summarized text")],
                    title="Text summarizer",
                    )
demo.launch()
