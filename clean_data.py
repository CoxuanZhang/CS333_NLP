"""
Author: Coco Zhang
Date: 12/12/2025
Description: Cleaning BBC News and the New Humanitarian datasets for final project

"""

import csv
import re
from collections import Counter
import pandas as pd

INSTITUTION_REGEX = r"""
(?ix)
\b(
    united\s+nations|u\.?n\.?|
    security\s+council|
    nato|
    european\s+union|eu|
    g7|g20|
    international\s+court|
    international\s+criminal\s+court|icc|
    world\s+bank|
    imf|
    aid\s+agency|ngo|relief\s+group|
    peacekeepers|
    sanctions|
    embassy|ambassador|
    foreign\s+ministry|foreign\s+secretary
)\b
"""

COUNTRY_REGEX = r"""
    (?ix)
    (?P<country>
    # Africa
    sudan|south\s*sudan|darfur|
    somalia|ethiopia|eritrea|niger|chad|
    uganda|kenya|nigeria|
    zimbabwe|south\s+africa|
    c[oô]te\s+d['’]ivoire|ivory\s+coast|
    democratic\s+republic\s+of\s+the\s+congo|dr\s+congo|drc|congo|

    # Middle East
    israel|palestine|gaza|west\s+bank|
    iran|iraq|syria|lebanon|
    saudi\s+arabia|yemen|jordan|egypt|
    turkey|

    # South & Central Asia
    afghanistan|pakistan|kashmir|
    india|bangladesh|sri\s+lanka|
    nepal|

    # East & Southeast Asia
    china|taiwan|hong\s+kong|
    north\s+korea|south\s+korea|
    japan|
    myanmar|burma|
    indonesia|philippines|thailand|vietnam|
    cambodia|laos|

    # Europe & Russia
    russia|ukraine|belarus|
    european\s+union|eu|
    france|germany|italy|spain|poland|

    # Americas
    united\s+states|u\.?s\.?a?\.?|america|
    canada|
    mexico|
    brazil|venezuela|colombia|
    haiti|cuba|

    # Regions
    middle\s+east|
    south\s+asia|
    east\s+asia|
    africa|sub[-\s]?saharan\s+africa
    )
"""
AFFAIRS_REGEX = r"""
(?ix)
\b(
    war|conflict|crisis|
    invasion|occupation|
    airstrike|bombing|
    ceasefire|
    sanctions|
    diplomacy|talks|negotiations|
    military|troops|forces|
    foreign\s+policy|
    arms\s+deal|
    nuclear|
    refugees?|asylum|
    peace\s
)\b
"""
country_re = re.compile(COUNTRY_REGEX, re.VERBOSE | re.IGNORECASE)
institution_re = re.compile(INSTITUTION_REGEX, re.VERBOSE | re.IGNORECASE)
affairs_re = re.compile(AFFAIRS_REGEX, re.VERBOSE | re.IGNORECASE)

def is_global_affairs(text):
    return (
        country_re.search(text)
        and (institution_re.search(text) or affairs_re.search(text))
    )

def extract_mentions(text):
    text = text.lower()

    return {
        "country": [m.group("country") for m in country_re.finditer(text)],
        "institution": [m.group(0) for m in institution_re.finditer(text)],
        "affairs": [m.group(0) for m in affairs_re.finditer(text)],
    }

def clean_bbc_text(f = "bbc_articles.csv"):
    with open(f, 'r') as file:
        reader = csv.DictReader(file)
        bbc_politics = []
        bbc_intl = []
        for row in reader:
            if row['category'] == 'politics':
                bbc_politics.append(row['text'])
                #if re.search(keywords, row['text'].lower(),re.DOTALL| re.IGNORECASE | re.VERBOSE):
                if is_global_affairs(row["text"]):
                    bbc_intl.append(row['text'])
    return bbc_politics, bbc_intl

def clean_text(f="The_New_Humanitarian.csv"):
    with open(f, 'r') as file:
        reader = csv.DictReader(file)
        articles = []
        for row in reader:
            text = row ['Title'] + row['Text'] 
            articles.append(text)
    return articles

def main():
    # Load and clean BBC 2004-2005 dataset
    bbc_politics, bbc_intl = clean_bbc_text()
    print(f"Total politics articles: {len(bbc_politics)}")
    print(f"Total international articles with keywords: {len(bbc_intl)}")

    # Count topics:
    country_counts = Counter()
    institution_counts = Counter()
    affairs_counts = Counter()
    for row in bbc_intl:   # already filtered dataset
        mentions = extract_mentions(row)
        country_counts.update(mentions["country"])
        institution_counts.update(mentions["institution"])
        affairs_counts.update(mentions["affairs"])

    """
    print("Top countries:")
    for k, v in country_counts.most_common(15):
        print(f"{k:30s} {v}")

    print("\nTop institutions:")
    for k, v in institution_counts.most_common(15):
        print(f"{k:30s} {v}")

    print("\nTop affairs terms:")
    for k, v in affairs_counts.most_common(15):
        print(f"{k:30s} {v}")
    """
    # Load and clean the bbc dataset
    copy_bbc_articles = [article for article in bbc_intl]
    duplicates = []
    for article in bbc_intl:
        copy_bbc_articles.remove(article)
        if article in copy_bbc_articles:
            print("Removing duplicate article")
            duplicates.append(article)
    for dup in duplicates:
        bbc_intl.remove(dup)
    print(f"Total BBC international articles after removing duplicates: {len(bbc_intl)}")


    # Load and clean the new humanitarian dataset
    tnh_articles = clean_text()
    copy_tnh_articles = [article for article in tnh_articles]
    duplicates = []
    for article in tnh_articles:
        copy_tnh_articles.remove(article)
        if article in copy_tnh_articles:
            print("Removing duplicate article")
            duplicates.append(article)
    for dup in duplicates:
        tnh_articles.remove(dup)
    print(f"Total The New Humanitarian articles after removing duplicates: {len(tnh_articles)}")

    mixed_data = []
    for article in bbc_intl:
        mixed_data.append({"text": article, "label": "bbc"})
    for article in tnh_articles:
        mixed_data.append({"text": article, "label": "tnh"})
    # Combine datasets with labels:
    with open('final/mixed_data.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames = ["text", "label"])
        writer.writeheader()
        writer.writerows(mixed_data)

if __name__ == "__main__":
    main()