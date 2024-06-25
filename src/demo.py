import gradio as gr
from PIL import Image
import requests
import json
import os
from io import BytesIO
import base64
from pinecone import Pinecone
from rag import get_single_image_embedding, get_image

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
    pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pinecone.Index("pathology")

    image_embedding = get_single_image_embedding(image)
    print(f"Successfully created image embedding.")

    query_response = index.query(vector=image_embedding.flatten().tolist(), top_k=3, include_values=True, include_metadata=True)
    print(f"Received query response: {query_response}")
    similar_images = [get_image(result['metadata']['image']) for result in query_response['matches']]
    print(f"Similar images: {similar_images}")

    return similar_images

with gr.Blocks() as demo:
    gr.Markdown("# Stardust")
    gr.Markdown("A tool for finding similar microscopic images of biopsy samples.")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Microscopic Image", sources=["upload"])
            submit_button = gr.Button("Find Similar Images")
        
        with gr.Column():
            gallery = gr.Gallery(label="Similar Images")
    
    submit_button.click(find_similar_images, inputs=image_input, outputs=gallery)
if __name__ == "__main__":
    demo.launch(share=False, debug=True)
