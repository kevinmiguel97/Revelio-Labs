import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re

def import_data_source(source_path): 
    # Specify folder with data
    folder_path_original = source_path
    folders = os.listdir(folder_path_original)


    # Create empty dictionary of files
    final_files = {}
    for f in folders:
        if f != '.DS_Store':
            folder_path = folder_path_original + '/' + f
            files = []
            print(f'Folder: {f} IMPORTED')
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".CSV") or file_name.endswith("csv"):
                    file_path = os.path.join(folder_path, file_name)
                    df = pd.read_csv(file_path)
                    files.append(df)

            # Concatenating files        
            df = pd.concat(files)

            # Add final file to dictionary
            final_files[str(f)] = df

    return final_files

# --------------------------------------------------------------------------------------------------------

def create_word_cloud(data, column, sentiment='negative'):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from wordcloud import WordCloud
    import random

    # Filter data for specified companies
    selected_companies = ['Microsoft Corp.', 'Alphabet, Inc.', 'Apple, Inc.', 'Meta Platforms, Inc.', 'Amazon.com, Inc.', 'Netflix, Inc.']
    df = data[data['company'].isin(selected_companies)]

    # Filter for time of interest
    df = df[(df['review_date_time'] >= '2022-12-01') & (df['review_date_time'] <= '2023-12-31')]

    # Your DataFrame 'df' contains the reviews
    reviews = df[column].dropna().str.lower().str.split()

    manual_list = ['amazon', 'will', 'job', 'time', 'employee', 'lot', 'mamager', 'con']

    # Remove stopwords
    filtered_reviews = [
        ' '.join([word.strip().upper() for word in review if (word not in ENGLISH_STOP_WORDS and word not in manual_list)])
        for review in reviews
    ]

    # Concatenate the filtered reviews into a single string
    text = ' '.join(filtered_reviews)

    # Function to generate random red color
    def random_red_color(*args, **kwargs):
        return f"rgb(255, {random.randint(0, 100)}, {random.randint(0, 100)})"
    
    # Function to generate random green color
    def random_green_color(*args, **kwargs):
        return f"rgb({random.randint(0, 100)}, 255, {random.randint(0, 100)})"

    if sentiment == 'positive':
        # Generate Word Cloud with different red tones
        wordcloud = WordCloud(width=600, height=510, background_color='white', max_words=20, collocations=True, color_func=random_green_color).generate(text)
    elif sentiment == 'negative':
        # Generate Word Cloud with different green tones
        wordcloud = WordCloud(width=600, height=510, background_color='white', max_words=20, collocations=False, color_func=random_red_color).generate(text)


    # Display the Word Cloud 
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save plot 
    if sentiment == 'negative':
        plt.savefig('plots/wordcloud_red.png')
    elif sentiment == 'positive':
        plt.savefig('plots/wordcloud_green.png')

    plt.show()
