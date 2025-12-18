This is Coco's CS333 Natural Language Processing Final Project: classifying and extracting features
from humanitarian journalism from different news outlets. I'm a sophomore at Wellesley College and the project works with the BBC News 2004-2005
dataset and a manually selected collection of articles published 2004-2005 on the New Humanitarian.

This is a guide to running the code:

The Gradescope submission is made up of the following components:
1. load_bbc.py: run to download the BBC News 2004-2005 dataset from UCD open repository 
2. The_New_Humanitarian.csv: csv file of articles from the new humanitarian (tnh) 2004 - 2005
3. clean_data.py: run to clean data from bbc and tnh and combine them in a labeled csv, 'mixed_data.csv'
4. part a.py: training and data visualisation for the tf-idf-based models on outlet categorisation
5. part b.py: semantic role labelling and data visualisation for the articles from both outlets

It's recommended to use a python 3.12 environment, and please install the following packages 
through your terminal prior to running any python code:
//pip install my-package
1. pandas (pd)
2. scikit-learn (sklearn)
3. spacy
4. plotly

Common issue:
1. If there's issue with opening a local csv file, check where the file is saved. In my local environment, I saved some 
documents under a folder called "final", which may create a relative path different from the one your using. Try ctrl+f
for "final" in the document, and delete it when appropriate.




