"""
Author: Coco Zhang
Date: 12/3/2025
Description: Importing and loading BBC News dataset for final project

"""

import requests
import zipfile
import os
import pandas as pd

def download_and_load_bbc():
    """Download and load BBC dataset from UCD repository."""
    
    # Download
    print("Downloading BBC dataset...")
    url = 'http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip'
    response = requests.get(url)
    
    with open('bbc-fulltext.zip', 'wb') as f:
        f.write(response.content)
    print("Downloaded: bbc-fulltext.zip")
    
    # Extract
    print("\nExtracting...")
    with zipfile.ZipFile('bbc-fulltext.zip', 'r') as zip_ref:
        zip_ref.extractall('bbc_data')
    print("Extracted:bbc_data")
    """
    # Debug: Show directory structure
    print("\n" + "="*60)
    print("DIRECTORY STRUCTURE:")
    print("="*60)
    for root, dirs, files in os.walk('bbc_data'):
        level = root.replace('bbc_data', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        sub_indent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files
            print(f'{sub_indent}{file}')
        if len(files) > 3:
            print(f'{sub_indent}... and {len(files)-3} more files')
    """
    # Load articles
    
    articles = []
    labels = []
    base_dir = 'bbc_data'
    print(f"Loading Articles from directory {base_dir}:")
    
    # The structure is usually: bbc_data/bbc/category/*.txt
    # Let's find it
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        for subitem in os.listdir(item_path):
            subitem_path = os.path.join(item_path, subitem)
            if os.path.isdir(subitem_path):
                txt_files = [f for f in os.listdir(subitem_path) if f.endswith('.txt')]
                if txt_files:
                    category = subitem
                    print(f"Loading category: {category} ({len(txt_files)} files)")
                    for filename in txt_files:
                        filepath = os.path.join(subitem_path, filename)
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read().strip()
                            if text:
                                articles.append(text)
                                labels.append(category)   
    print(f"\nLoaded {len(articles)} articles total")
    
    if len(articles) == 0:
        print("\nERROR: No articles were loaded!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': articles,
        'category': labels
    })
    
    return df

# Run it
df = download_and_load_bbc()

if df is not None and len(df) > 0:
    print(f"Total articles in dataframe: {len(df)}")
    # Save to CSV
    df.to_csv('bbc_articles.csv', index=False)
    print("\n Saved dataset to bbc_articles.csv")

else:
    print("\nFailed to load dataset.")