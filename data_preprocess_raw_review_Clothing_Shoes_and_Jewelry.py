# !pip install datasets
import datasets
import pandas as pd
datasets.logging.set_verbosity_error()
from datasets import load_dataset

# load the dataset
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Clothing_Shoes_and_Jewelry", trust_remote_code=True)
print('load dataset done')

# Convert to DataFrame
reviews_df = pd.DataFrame(dataset['full'])
print('convert to DataFrame done')

# save the data as a feather file
reviews_df.to_feather('review_data.feather')
print('save as feather done')

# print the column names and the first row of the data
# print(reviews_df.columns)
# print(reviews_df.iloc[0])

# load metadata json file
import json

with open('raw_meta_dresses.json') as f:
    metadata = json.load(f)
print('load metadata done')

# Convert to DataFrame
metadata_df = pd.DataFrame(metadata)
print('convert to DataFrame done')

# save the data as a feather file
metadata_df.to_feather('metadata.feather')
print('save as feather done')

# merge metadata and reviews data
combined_df = pd.merge(metadata_df, reviews_df, on='parent_asin', how='inner')

# save the combined data as a json file
combined_df.to_json('combined_data.json', orient='records', lines=True)
print('save as json done')

# save the combined data as a feather file
combined_df.to_feather('combined_data.feather')

# for data in metadata, want to concatenate same parent_asin data from dataset to metadata
# parent_asin is the unique identifier for each product
# the whole data will be saved as a json file
