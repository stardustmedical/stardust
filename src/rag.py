import os
from pinecone import Pinecone
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import requests
from typing import List
from io import BytesIO
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "pathology"
index = pc.Index(index_name)

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model_info(model_ID, device):
    model = CLIPModel.from_pretrained(model_ID).to(device)
    processor = CLIPProcessor.from_pretrained(model_ID)
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    return model, processor, tokenizer


model_ID = "openai/clip-vit-base-patch32"
model, processor, tokenizer = get_model_info(model_ID, device)
data = pd.read_json("examples.jsonl", lines=True)
model.to(device)


def get_single_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=77)
    text_embeddings = model.get_text_features(**inputs)
    embedding_as_np = text_embeddings.cpu().detach().numpy()
    return embedding_as_np


def get_all_text_embeddings(df, text_col):
    df["text_embeddings"] = df[str(text_col)].apply(get_single_text_embedding)
    return df


def get_single_image_embedding(my_image) -> List:
    image = processor(text=None, images=my_image, return_tensors="pt")[
        "pixel_values"
    ].to(device)
    embedding = model.get_image_features(image)
    embedding_as_np = embedding.cpu().detach().numpy()
    return embedding_as_np


def get_all_images_embedding(df, img_column):
    df["img_embeddings"] = df[str(img_column)].apply(get_single_image_embedding)
    return df


def check_valid_URLs(image_URL):
    try:
        response = requests.get(image_URL)
        Image.open(BytesIO(response.content))
        return True
    except:
        return False


def get_image(image_url):
    print(f"Image url: {image_url}")
    response = requests.get(image_url)
    print(f"Response: {response}")

    if response.status_code == 200 and "image" in response.headers["Content-Type"]:
        try:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return image
        except:
            print(f"Error: Cannot identify image file from URL {image_url}")
            return None
    else:
        print(
            f"Error: Failed to retrieve image from URL {image_url} with status code {response.status_code}"
        )
        return None


data_df = pd.DataFrame(data)
image_data_df = get_all_text_embeddings(data_df, "description")

image_data_df["is_valid"] = image_data_df["image_url"].apply(check_valid_URLs)
image_data_df = image_data_df[image_data_df["is_valid"] == True]
image_data_df["image"] = image_data_df["image_url"].apply(get_image)
image_data_df = get_all_images_embedding(image_data_df, "image")
image_data_df.loc[:, "vector_id"] = image_data_df.index.astype(str)

image_data_df["vector_id"] = image_data_df.index
image_data_df["vector_id"] = image_data_df["vector_id"].apply(str)

final_metadata = []
for idx in range(len(image_data_df)):
    final_metadata.append(
        {
            "ID": idx,
            "description": image_data_df.iloc[idx].description,
            "image": image_data_df.iloc[idx].image_url,
        }
    )
image_IDs = image_data_df.vector_id.tolist()
image_embeddings = [
    embedding
    for sublist in image_data_df.img_embeddings.tolist()
    for embedding in sublist
]
# Create the single list of dictionary format to insert
data_to_upsert = list(zip(image_IDs, image_embeddings, final_metadata))
index.upsert(vectors=data_to_upsert)
print(index.describe_index_stats())
