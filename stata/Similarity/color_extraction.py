import pandas as pd
import cv2
import numpy as np
import requests
import json

def load_image_from_url(url):
    if not url:
        return None
    resp = requests.get(url)
    image = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def compute_normalized_histogram(image, bins=256):
    if image is None:
        return None
    hist = []
    for i in range(3):  # For each color channel
        channel_hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        cv2.normalize(channel_hist, channel_hist, norm_type=cv2.NORM_L2)
        hist.append(channel_hist)
    return hist

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    if hist1 is None or hist2 is None:
        return None  # Return None if any histogram is missing
    similarity = sum(cv2.compareHist(hist1[i], hist2[i], method) for i in range(3)) / 3
    return similarity

def extract_first_image_url(image_list_str):
    try:
        image_list = json.loads(image_list_str.replace("'", '"'))
        if image_list:
            return image_list[0]
        else:
            return None
    except json.JSONDecodeError:
        return None

def calculate_similarity(row):
    image_cgi = load_image_from_url(row['cgi_image_url'])
    image_fgi = load_image_from_url(row['fgi_image_url'])
    hist_cgi = compute_normalized_histogram(image_cgi)
    hist_fgi = compute_normalized_histogram(image_fgi)
    return compare_histograms(hist_cgi, hist_fgi)

def main():
    df = pd.read_csv('df_review.csv')
    df['fgi_image_url'] = df['fgi_images'].apply(extract_first_image_url)
    df['cgi_image_url'] = df['cgi_images'].apply(extract_first_image_url)  # Adjust column name if different
    df['color_similarity'] = df.apply(calculate_similarity, axis=1)
    df.to_csv('updated_df_review.csv', index=False)
    print("Color similarity calculation completed. Results saved to 'updated_df_review.csv'.")

if __name__ == "__main__":
    main()