import gradio as gr
from PIL import Image
import requests
import json
import os
from io import BytesIO
import base64
from dotenv import load_dotenv

load_dotenv()

from exa_py import Exa

exa = Exa(api_key=os.getenv("EXA_API_KEY"))


def search(query: str): 
    result = exa.search_and_contents(
        query=query, 
        type="neural",
        num_results=5,
        text=True,
        category="research paper"
    )
    print(result)

    return result


url = "https://api.fireworks.ai/inference/v1/chat/completions"
payload = {
  "model": "accounts/fireworks/models/llava-yi-34b",
  "max_tokens": 512,
  "top_p": 1,
  "top_k": 40,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "temperature": 0.6,
  "messages": []
}
headers = {
  "Accept": "application/json",
  "Content-Type": "application/json",
  "Authorization": f"Bearer {os.getenv("EXA_API_KEY")}"
}    
def generate_description(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    payload = {
        "model": "accounts/fireworks/models/llava-yi-34b",
        "max_tokens": 512,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the content of this image in detail. Be succinct"
                    },
                    {
                        "type": "image",
                        "image": image_data
                    }
                ]
            }
        ]
    }


    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    return response.json()


def find_similar_images(image):
    # TODO: Add logic to find similar images
    similar_images = [image, image, image]  # Placeholder for now
    return similar_images

with gr.Blocks() as demo:
    gr.Markdown("# Pathopaedia")
    gr.Markdown("""
    Pathopaedia is a tool for identifying and describing pathological changes in microscopic images of tissue samples.
    """)
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Microscopic Image", sources=["upload", "clipboard"])
            submit_button = gr.Button("Find Similar Images")
        
        with gr.Column():
            gallery = gr.Gallery(label="Similar Images")
    
    submit_button.click(find_similar_images, inputs=image_input, outputs=gallery)

if __name__ == "__main__":
    demo.launch(share=False)
    # img = Image.open("/Users/leandermarkisch/Downloads/histo_image.jpg")
    # image_description = generate_description(img)
    # print(image_description)

    # search_result = search(query="pathology ai diseases")
