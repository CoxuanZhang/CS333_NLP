import csv
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import re
import spacy
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

# Keep parser disabled for speed
nlp = spacy.load("en_core_web_sm", disable=["parser"])
# Entities to mask: common leak sources
MASK_TYPES = {"PERSON", "ORG", "GPE"}

# Task A: Outlet classification (BBC vs TNH)
# Clean Data Again
def scrub_text(s):
    # remove outlet names
    s = re.sub(r"\bBBC\b|\birin\b|\bbbc\b|\bIRIN\b", "", s, flags=re.I)
    s = re.sub(r"\bThe New Humanitarian\b|\bTNH\b", "", s, flags=re.I)
    # remove bylines with author names (potential style leakage)
    s = re.sub(r"\bBy\s+[A-Z][a-z]+\b", "", s)
    return s

# NER-Masked Model: TF-IDF + Logistic Regression
def mask_ents(text: str) -> str:
    doc = nlp(text)
    out = []
    last = 0
    for ent in doc.ents:
        if ent.label_ in MASK_TYPES:
            out.append(text[last:ent.start_char])
            out.append(f"[{ent.label_}]")
            last = ent.end_char
    out.append(text[last:])
    return " ".join("".join(out).split())

# Topic-based Masking:

BBC_PHRASES = ["prime minister", "foreign secretary", "general election"]
TNH_PHRASES = ["aid worker", "security council"]

BBC_LEMMAS = {
    "election", "party", "tory", "tories", "parliament",
    "chancellor", "secretary", "minister", "labour", "liberal", "immigration",
    "british", "uk"
}
TNH_LEMMAS = {
    "refugee", "displace", "aid", "camp", "humanitarian", "asylum",
    "hunger", "famine", "cholera", "relief"
}

def mask_genre(text: str) -> str:
    s = text
    for p in BBC_PHRASES:
        s = re.sub(re.escape(p), " [BBC] ", s, flags=re.I)
    for p in TNH_PHRASES:
        s = re.sub(re.escape(p), " [TNH] ", s, flags=re.I)

    doc = nlp(s)
    out = []
    for tok in doc:
        lemma = tok.lemma_.lower() if tok.lemma_ else tok.text.lower()
        if lemma in BBC_LEMMAS:
            out.append("[BBC]" + tok.whitespace_)
        elif lemma in TNH_LEMMAS:
            out.append("[TNH]" + tok.whitespace_)
        else:
            out.append(tok.text_with_ws)
    return "".join(out)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True, #ignore case
        stop_words="english", #provided stop words list for English
        ngram_range=(1, 2), #unigrams and bigrams
        max_df=0.7, #min_df=3
        sublinear_tf=True #replace tf with 1 + log(tf)
    )),
    ("clf", LogisticRegression(
        #max_iter=1000,
        class_weight="balanced",   # safe default in case slight imbalance
        warm_start=False # re-fit from scratch each time esp. with 5-fold CV
    ))
])

def batch_dataset(file, batches = 5):
    with open(file, "r") as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
        random.shuffle(data)
        batch_size = int(len(data)/batches)
        for i in range(batches):
            with open (f"batch{i+1}.csv", "w") as batch_f:
                writer = csv.DictWriter(batch_f, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in data[i*batch_size:(i+1)*batch_size]:
                    writer.writerow(row)
    return [f"batch{i}.csv" for i in range(1, 1+batches)]

def run_model(dfs, top_n = 10,model: int = 1):
    performance_f1 = {'macro-average':[],'bbc':[], 'tnh':[]} # dict
    # Preprocess each dataframe in-place (scrub and optional masking)
    for idx, df in enumerate(dfs):
        print(f"Preprocessing dataframe {idx+1}/{len(dfs)}")
        key = 'text'
        clean = df[key].astype(str).apply(scrub_text)
        if model > 1:
            key = 'text_clean'
            # mask if NER:
            df[key] = clean.apply(mask_ents)
            if model > 2:
                # mask if topic-based:
                df[key] = df[key].apply(mask_genre)
    for i in range(len(dfs)):
        print(f"Starting fold {i+1}/{len(dfs)}")
        test_df = dfs[i]
        train_dfs = dfs[:i] + dfs[i+1:]
        train_df = pd.concat(train_dfs, ignore_index=True)

        X_train = train_df[key].tolist()
        y_train = train_df["label"].tolist()
        X_test = test_df[key].tolist()
        y_test = test_df["label"].tolist()

        pipe.fit(X_train, y_train)
        
        top_features = {'bbc':[],'tnh':[]}
        # extract top tf-idf features and coefficients at the last fold
        if (i+1) == len(dfs):
            vec = pipe.named_steps["tfidf"]
            clf = pipe.named_steps["clf"]
            feature_names = vec.get_feature_names_out()
            coefs = clf.coef_[0]
            Xv_train = vec.transform(X_train)
            # per-class document frequency in the training set (uses same tokenization)
            mask_bbc = np.array(y_train) == 'bbc'
            mask_tnh = np.array(y_train) == 'tnh'
            df_bbc = np.asarray((Xv_train[mask_bbc, :] > 0).sum(axis=0)).ravel()
            df_tnh = np.asarray((Xv_train[mask_tnh, :] > 0).sum(axis=0)).ravel()

            inds_pos = np.argsort(coefs)[-top_n:][::-1]
            inds_neg = np.argsort(coefs)[:top_n]

            #Top TF-IDF features (TNH-leaning)
            for j in inds_pos:
                term = feature_names[j]
                dic = {"term": f"{term:20s}", "coef":round(coefs[j],4), "bbc": int(df_bbc[j]), "tnh_df":int(df_tnh[j])}
                top_features['tnh'].append(dic)

            #Top TF-IDF features (BBC-leaning)
            for j in inds_neg:
                term = feature_names[j]
                dic = {"term": f"{term:20s}", "coef":round(coefs[j],4), "bbc": int(df_bbc[j]), "tnh_df":int(df_tnh[j])}
                top_features['bbc'].append(dic)
        y_pred = pipe.predict(X_test)

        print(f"Fold {i+1} Classification Report:")
        perf = classification_report(y_test, y_pred, output_dict=True)
        performance_f1['macro-average'].append(perf['macro avg']['f1-score'])
        performance_f1['bbc'].append(perf['bbc']['f1-score'])
        performance_f1['tnh'].append(perf['tnh']['f1-score'])
        print(classification_report(y_test, y_pred))
        print(f"Fold {i+1} Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    # prepare for visualisation
    return performance_f1, top_features

# functions for tf-idf feature data visualisation
def to_long_df(model_dict, model_name):
    rows = []
    for label in ["bbc", "tnh"]:
        for d in model_dict[label]:
            rows.append({
                "model": model_name,
                "class": label,
                "term": d["term"].strip(),
                "coef": float(d["coef"]),
                "bbc_df": int(d["bbc"]),
                "tnh_df": int(d["tnh_df"]),
            })
    return pd.DataFrame(rows)

def plot_features(df, model_name, topn=10):
    d = df[df["model"] == model_name].copy()

    bbc = d[d["class"]=="bbc"].sort_values("coef").head(topn)
    tnh = d[d["class"]=="tnh"].sort_values("coef", ascending=False).head(topn)
    plot_df = pd.concat([bbc, tnh]).copy()
    plot_df = plot_df.sort_values("coef")  # negatives first

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["term"], plot_df["coef"])
    plt.axvline(0, linewidth=1)
    plt.title(model_name)
    plt.xlabel("TF-IDF coefficient")
    plt.tight_layout()
    plt.show()

def plot_rank_paired_bars(df, model_name, topn=10):
    d = df[df["model"] == model_name].copy()

    bbc = (d[d["class"]=="bbc"]
           .sort_values("coef")  # most negative first
           .head(topn)
           .reset_index(drop=True))
    tnh = (d[d["class"]=="tnh"]
           .sort_values("coef", ascending=False)  # most positive first
           .head(topn)
           .reset_index(drop=True))

    # ensure same length
    n = min(len(bbc), len(tnh))
    bbc, tnh = bbc.iloc[:n], tnh.iloc[:n]

    ranks = np.arange(1, n+1)

    # y positions (one row per rank)
    y = np.arange(n)

    # Labels for each side
    left_labels  = [f"{r}. {t}" for r, t in zip(ranks, bbc["term"].astype(str))]
    right_labels = [f"{r}. {t}" for r, t in zip(ranks, tnh["term"].astype(str))]

    plt.figure(figsize=(12, 6))

    # left: BBC (negative)
    plt.barh(y, bbc["coef"].values, alpha=0.9)
    # right: TNH (positive)
    plt.barh(y, tnh["coef"].values, alpha=0.9)

    plt.axvline(0, linewidth=1)

    # Put BBC labels on left, TNH labels on right
    # Use two y-axes so labels don’t collide
    ax = plt.gca()
    ax.set_yticks(y)
    ax.set_yticklabels(left_labels, fontsize=10)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y)
    ax2.set_yticklabels(right_labels, fontsize=10)

    ax.set_xlabel("LogReg coefficient (TF-IDF)")
    ax.set_ylabel("BBC top terms by rank")
    ax2.set_ylabel("TNH top terms by rank")

    plt.title(f"{model_name}: rank-matched top features (BBC vs TNH)")
    plt.tight_layout()
    plt.show()

def top_terms_union(df, cls, k=6):
    terms = set()
    for model in df["model"]:
        d = df[(df["model"] == model) & (df["class"] == cls)]
        if cls == "bbc":
            top = d.sort_values("coef").head(k)["term"]
        else:
            top = d.sort_values("coef", ascending=False).head(k)["term"]
        terms |= set(top)
    return sorted(terms)

model_order = ["TF-IDF Baseline","NER-Masked","Topic-Masked"]

def pivot_for_slope(df, cls, terms, order = model_order):
    d = df[(df["class"] == cls) & (df["term"].isin(terms))]
    piv = d.pivot(index="term", columns="model", values="coef")
    piv = piv.reindex(columns=order)
    return piv

def plot_trend(piv, title, color): # plot trends for pivot features
    models = list(piv.columns)  # now guaranteed correct order
    x = range(len(models))

    plt.figure(figsize=(9, 5))

    for term, row in piv.iterrows():
        y = row.values.astype(float)
        plt.plot(x, y, marker="o", linewidth=2, alpha=0.85)
        if not pd.isna(y[0]):
            plt.text(x[0] - 0.05, y[0], term, ha="right", va="center", fontsize=9)
        if not pd.isna(y[-1]):
            plt.text(x[-1] + 0.05, y[-1], term, ha="left", va="center", fontsize=9)

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, models)
    plt.ylabel("TF-IDF Coefficient Score")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    file_lst = batch_dataset("final/mixed_data.csv")
    dfs = [pd.read_csv(fn) for fn in file_lst]
    model_label = {1:"TF-IDF Baseline", 2:"NER-Masked", 3:"Topic-Masked"}
    # Baseline Model: TF-IDF + Logistic Regression
    # Advanced Model: Masked with NER (and Topic)
    perf_1, features_1= run_model(dfs, model=1)
    perf_2, features_2 = run_model(dfs, model=2)
    perf_3, features_3 = run_model(dfs, model=3)

    # Visualisation: build a tidy DataFrame with one row per fold
    rows = []
    mapping = [(perf_1, model_label[1]), (perf_2, model_label[2]), (perf_3, model_label[3])]
    for perf, mlabel in mapping:
        for metric, scores in perf.items():
            for s in scores:
                rows.append({"model": mlabel, "metric": metric, "f1": s})

    df_plot = pd.DataFrame(rows)
    # Use a boxplot to show distribution across folds for each model and metric
    fig = px.box(df_plot, x="model", y="f1", color="metric", points="all",
                 title="Model Performance Comparison (F1-Score across folds)")
    fig.update_layout(height=500)
    fig.show()

    # tf-idf feature analysis and visualisation
    df1 = to_long_df(features_1, "TF-IDF Baseline")
    df2 = to_long_df(features_2, "NER-Masked")
    df3 = to_long_df(features_3, "Topic-Masked")
    df_feature = pd.concat([df1, df2, df3], ignore_index=True)

    for model in model_label.values():
        plot_rank_paired_bars(df_feature, model, topn=10)
    
    # show top terms trend between models to show the effectiveness of masking
    
    bbc_terms = top_terms_union(df_feature, "bbc")
    tnh_terms = top_terms_union(df_feature, "tnh")

    bbc_piv = pivot_for_slope(df_feature, "bbc", bbc_terms)
    tnh_piv = pivot_for_slope(df_feature, "tnh", tnh_terms)

    plot_trend(bbc_piv,title="Top BBC Terms Before vs After Masking",color="tab:blue")
    plot_trend(tnh_piv,title="Top TNH Terms Before vs After Masking",color="tab:red")

if __name__ == "__main__":
    main()