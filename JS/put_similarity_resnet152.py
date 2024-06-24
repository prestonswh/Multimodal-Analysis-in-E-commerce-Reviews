'''


image_model: 내가 원하는 image embedding model
text_model: 내가 원하는 text embedding model
'''

import os
import ast
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import torch
from scipy.spatial.distance import cosine
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms

# Load ResNet50 model for image similarity
device = "cuda" if torch.cuda.is_available() else "cpu"

'''
아래 두 줄만 model 입력해주면 됨.
'''

image_model = models.resnet152(pretrained=True).to(device)
text_model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv('/Users/ojeongsig/code_fgi/only_cgi.csv') #CGI만 있는 데이터 (text 포함 X)
output_path='/Users/ojeongsig/code_fgi/result_similarity.csv'



image_model.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the sentence-transformer model for text similarity


# Helper functions
def open_image(file_path=None, fallback_url=None):
    try:
        if file_path and os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
        elif fallback_url:
            response = requests.get(fallback_url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            print(f"No such file or directory: '{file_path}' and no fallback URL provided.")
            return None
        return img
    except Exception as e:
        print(f"Error opening image {file_path}: {e}")
        return None

def get_image_similarity(image1_path, image2_url, model, preprocess, device):
    try:
        # Load images
        image1 = open_image(fallback_url=image1_path)
        image2 = open_image(fallback_url=image2_url)

        if image1 is None or image2 is None:
            return 0

        # Preprocess images
        image1_tensor = preprocess(image1).unsqueeze(0).to(device)
        image2_tensor = preprocess(image2).unsqueeze(0).to(device)

        # Get image embeddings
        with torch.no_grad():
            image1_embedding = model(image1_tensor).cpu().numpy().flatten()
            image2_embedding = model(image2_tensor).cpu().numpy().flatten()

        # Compute cosine similarity
        cosine_sim = 1 - cosine(image1_embedding, image2_embedding)
        normalized_similarity = (cosine_sim + 1) / 2
        similarity_score = normalized_similarity * 100
        similarity_score = min(100, similarity_score)

    except Exception as e:
        print(f"Error calculating image similarity: {e}")
        similarity_score = 0

    return similarity_score

def get_text_similarity(text1, text2, model):
    try:
        # Get text embeddings
        embeddings = model.encode([text1, text2])
        text1_embedding, text2_embedding = embeddings[0], embeddings[1]

        # Compute cosine similarity
        similarity_score = cosine_similarity([text1_embedding], [text2_embedding])[0][0] * 100
        similarity_score = min(100, similarity_score)

    except Exception as e:
        print(f"Error calculating text similarity: {e}")
        similarity_score = 0

    return similarity_score

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def get_image_text_pair_similarity(cgi_image_path, cgi_text, fgi_image_url, fgi_text, image_model, preprocess, text_model, device):
    try:
        # Load CGI image
        cgi_image = open_image(fallback_url=cgi_image_path)
        fgi_image = open_image(fallback_url=fgi_image_url)

        if cgi_image is None or fgi_image is None:
            return 0

        # Preprocess image and text pairs
        cgi_image_tensor = preprocess(cgi_image).unsqueeze(0).to(device)
        fgi_image_tensor = preprocess(fgi_image).unsqueeze(0).to(device)

        # Get text embeddings
        text_embeddings = text_model.encode([cgi_text, fgi_text])
        cgi_text_embedding, fgi_text_embedding = text_embeddings[0], text_embeddings[1]

        # Print text embeddings dimensions
        print(f"CGI Text Embedding Dimension: {cgi_text_embedding.shape}")
        print(f"FGI Text Embedding Dimension: {fgi_text_embedding.shape}")

        # Normalize text embeddings
        cgi_text_embedding = normalize_embedding(cgi_text_embedding)
        fgi_text_embedding = normalize_embedding(fgi_text_embedding)

        # Get image embeddings
        with torch.no_grad():
            cgi_image_embedding = image_model(cgi_image_tensor).cpu().numpy().flatten()
            fgi_image_embedding = image_model(fgi_image_tensor).cpu().numpy().flatten()

        # Print image embeddings dimensions
        print(f"CGI Image Embedding Dimension: {cgi_image_embedding.shape}")
        print(f"FGI Image Embedding Dimension: {fgi_image_embedding.shape}")

        # Normalize image embeddings
        cgi_image_embedding = normalize_embedding(cgi_image_embedding)
        fgi_image_embedding = normalize_embedding(fgi_image_embedding)

        # Concatenate normalized image and text embeddings
        cgi_combined_embedding = np.concatenate((cgi_image_embedding, cgi_text_embedding))
        fgi_combined_embedding = np.concatenate((fgi_image_embedding, fgi_text_embedding))

        # Compute cosine similarity
        cosine_sim = 1 - cosine(cgi_combined_embedding, fgi_combined_embedding)
        normalized_similarity = (cosine_sim + 1) / 2
        similarity_score = normalized_similarity * 100
        similarity_score = min(100, similarity_score)

    except Exception as e:
        print(f"Error calculating image-text pair similarity: {e}")
        similarity_score = 0

    return similarity_score


# Load the CSV file


df_sample = df

# Initialize lists to store similarity results
image_similarity_list = []
text_similarity_list = []
pair_similarity_list = []

# Calculate similarities and store in the DataFrame
for index, row in tqdm(df_sample.iterrows(), total=df_sample.shape[0]):
    cgi_image_path = ast.literal_eval(row['cgi_images'])[0] if row['cgi_images'] != '[]' else None
    fgi_image_url = ast.literal_eval(row['fgi_images'])[0] if row['fgi_images'] != '[]' else None
    review_text = row['review_text']
    feature_text = row['features']
    
    if pd.notna(cgi_image_path):
        # Calculate similarities
        image_similarity = get_image_similarity(cgi_image_path, fgi_image_url, image_model, preprocess, device)
        text_similarity = get_text_similarity(review_text, feature_text, text_model)
        pair_similarity = get_image_text_pair_similarity(cgi_image_path, review_text, fgi_image_url, feature_text, image_model, preprocess, text_model, device)
        
        # Append similarity scores to lists
        image_similarity_list.append(image_similarity)
        text_similarity_list.append(text_similarity)
        pair_similarity_list.append(pair_similarity)

# Add the similarity results to the DataFrame
df_sample.loc[:, 'image_similarity'] = image_similarity_list
df_sample.loc[:, 'text_similarity'] = text_similarity_list
df_sample.loc[:, 'pair_similarity'] = pair_similarity_list

# Save the DataFrame with similarity results to a new CSV file
sample_csv_path = output_path
df_sample.to_csv(sample_csv_path, index=False)

print(f"Sample data with similarities saved to {sample_csv_path}")
