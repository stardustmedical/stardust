import gradio as gr
import os
import multiprocessing
from pinecone import Pinecone
from rag import get_single_image_embedding, get_image

from dotenv import load_dotenv

load_dotenv()

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
    demo.launch(share=True, debug=True)
