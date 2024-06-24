import pandas as pd
from datetime import datetime

def load_and_merge_csv(csv1_path, csv2_path, columns_to_merge):
    # Load the first CSV file - CGI 포함된 데이터
    csv1 = pd.read_csv(csv1_path)

    # Load the second CSV file - 초기 원본데이터
    csv2 = pd.read_csv(csv2_path)

    # Set 'Unnamed: 0' as the index for the first DataFrame
    csv1.set_index('Unnamed: 0', inplace=True)

    # Merge the specified columns into csv2 based on the 'Unnamed: 0' index
    csv2 = csv2.merge(csv1[columns_to_merge], how='left', left_index=True, right_index=True)

    # Function to fill values only for rows below the first occurrence of non-missing values
    def fill_below_first_non_na(df, columns):
        for col in columns:
            mask = df[col].notna()
            first_non_na_idx = mask.idxmax() if mask.any() else None
            if first_non_na_idx:
                df.loc[first_non_na_idx+1:, col] = df.loc[first_non_na_idx, col]
        return df

    # Apply the function to each group of 'product_id'
    csv2 = csv2.groupby('product_id', group_keys=False).apply(fill_below_first_non_na, columns=columns_to_merge)

    # Fill NaN values with 0
    csv2[columns_to_merge] = csv2[columns_to_merge].fillna(0)

    return csv2

def process_data(df):
    # Identify duplicates based on product_id and review_date
    duplicates = df[df.duplicated(subset=['product_id', 'review_date'], keep=False)]

    # Print duplicates if they exist
    if not duplicates.empty:
        print("Duplicates found:")
        print(duplicates)
    else:
        print("No duplicates found.")

    # Drop duplicates based on product_id and review_date
    df = df.drop_duplicates(subset=['product_id', 'review_date'])
    
    # Convert review_date to the specified datetime format
    df['datetime'] = pd.to_datetime(df['review_date']).dt.strftime('%d%b%Y %H:%M:%S')
    
    # Extract month and year from the datetime
    df['mon'] = pd.to_datetime(df['review_date']).dt.month
    df['year'] = pd.to_datetime(df['review_date']).dt.year
    
    return df

def main():
    # Define file paths and columns to merge
    '''
    cv1: CGI만 있고, 구하고자 하는 열이 추가된 데이터
    cv2: 원본데이터
    output file: 최종결과값 이름 바꿔주기!
    '''
    csv1_path = '/Users/ojeongsig/code_fgi/06_14.csv'
    csv2_path = '/Users/ojeongsig/code_fgi/df_review_backup.csv'
    columns_to_merge = ['image_similarity', 'text_similarity', 'pair_similarity']
    output_file = '/Users/ojeongsig/code_fgi/result_processed.csv'
    
    # Load and merge CSV files
    merged_df = load_and_merge_csv(csv1_path, csv2_path, columns_to_merge)
    
    # Process data
    processed_df = process_data(merged_df)
    
    # Save processed data
    processed_df.to_csv(output_file, index=False)
    
    print("Data processing complete. Files saved.")

if __name__ == "__main__":
    main()
