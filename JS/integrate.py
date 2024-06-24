'''
싹다 통합 (sentiment 제외) --> 내일 test해볼 예정

image_model = models.resnet152(pretrained=True).to(device)
text_model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv('/Users/ojeongsig/code_fgi/only_cgi.csv')
csv1_path = '/Users/ojeongsig/code_fgi/result_similarity.csv'
csv2_path = '/Users/ojeongsig/code_fgi/df_revie_backup.csv'
columns_to_merge = ['image_similarity', 'text_similarity', 'pair_similarity']
output_file = '/Users/ojeongsig/code_fgi/result_processed.csv'



input: only cgv만 있는 csv file + 원본 csv file 
hyper_parameter: image embedding model 이름, text embedding model 이름
output: 원본데이터에 image similarity, text similarity, image+text similarity 열 추가된 결과 나옴.

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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms

# Load device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Image and Text Model Initialization
image_model = models.resnet152(pretrained=True).to(device)
text_model = SentenceTransformer('all-MiniLM-L6-v2')
image_model.eval()

# Preprocessing for images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the data
df = pd.read_csv('/Users/ojeongsig/code_fgi/only_cgi.csv')
output_path = '/Users/ojeongsig/code_fgi/result_similarity.csv'

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

        # Normalize text embeddings
        cgi_text_embedding = normalize_embedding(cgi_text_embedding)
        fgi_text_embedding = normalize_embedding(fgi_text_embedding)

        # Get image embeddings
        with torch.no_grad():
            cgi_image_embedding = image_model(cgi_image_tensor).cpu().numpy().flatten()
            fgi_image_embedding = image_model(fgi_image_tensor).cpu().numpy().flatten()

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

# Calculate similarities and store in the DataFrame
df_sample = df
image_similarity_list = []
text_similarity_list = []
pair_similarity_list = []

for index, row in tqdm(df_sample.iterrows(), total=df_sample.shape[0]):
    cgi_image_path = ast.literal_eval(row['cgi_images'])[0] if row['cgi_images'] != '[]' else None
    fgi_image_url = ast.literal_eval(row['fgi_images'])[0] if row['fgi_images'] != '[]' else None
    review_text = row['review_text']
    feature_text = row['features']
    
    if pd.notna(cgi_image_path):
        image_similarity = get_image_similarity(cgi_image_path, fgi_image_url, image_model, preprocess, device)
        text_similarity = get_text_similarity(review_text, feature_text, text_model)
        pair_similarity = get_image_text_pair_similarity(cgi_image_path, review_text, fgi_image_url, feature_text, image_model, preprocess, text_model, device)
        
        image_similarity_list.append(image_similarity)
        text_similarity_list.append(text_similarity)
        pair_similarity_list.append(pair_similarity)

df_sample.loc[:, 'image_similarity'] = image_similarity_list
df_sample.loc[:, 'text_similarity'] = text_similarity_list
df_sample.loc[:, 'pair_similarity'] = pair_similarity_list

sample_csv_path = output_path
df_sample.to_csv(sample_csv_path, index=False)
print(f"Sample data with similarities saved to {sample_csv_path}")

# Additional processing and merging
def load_and_merge_csv(csv1_path, csv2_path, columns_to_merge):
    csv1 = pd.read_csv(csv1_path)
    csv2 = pd.read_csv(csv2_path)
    csv1.set_index('Unnamed: 0', inplace=True)
    csv2 = csv2.merge(csv1[columns_to_merge], how='left', left_index=True, right_index=True)

    def fill_below_first_non_na(df, columns):
        for col in columns:
            mask = df[col].notna()
            first_non_na_idx = mask.idxmax() if mask.any() else None
            if first_non_na_idx:
                df.loc[first_non_na_idx+1:, col] = df.loc[first_non_na_idx, col]
        return df

    csv2 = csv2.groupby('product_id', group_keys=False).apply(fill_below_first_non_na, columns=columns_to_merge)
    csv2[columns_to_merge] = csv2[columns_to_merge].fillna(0)
    return csv2

def process_data(df):
    duplicates = df[df.duplicated(subset=['product_id', 'review_date'], keep=False)]
    if not duplicates.empty:
        print("Duplicates found:")
        print(duplicates)
    else:
        print("No duplicates found.")
    df = df.drop_duplicates(subset=['product_id', 'review_date'])
    df['datetime'] = pd.to_datetime(df['review_date']).dt.strftime('%d%b%Y %H:%M:%S')
    df['mon'] = pd.to_datetime(df['review_date']).dt.month
    df['year'] = pd.to_datetime(df['review_date']).dt.year
    return df

def main():
    csv1_path = '/Users/ojeongsig/code_fgi/result_similarity.csv'
    csv2_path = '/Users/ojeongsig/code_fgi/df_revie_backup.csv'
    columns_to_merge = ['image_similarity', 'text_similarity', 'pair_similarity']
    output_file = '/Users/ojeongsig/code_fgi/result_processed.csv'
    
    merged_df = load_and_merge_csv(csv1_path, csv2_path, columns_to_merge)
    processed_df = process_data(merged_df)
    processed_df.to_csv(output_file, index=False)
    print("Data processing complete. Files saved.")

if __name__ == "__main__":
    main()
