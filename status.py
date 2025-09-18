"""
CORD-19 metadata exploration + Streamlit app

This single text file contains two separate Python scripts one after another. Save them as two files
in the same folder as your downloaded metadata.csv (or edit the METADATA_PATH variable).

Files contained:
  1) cord19_analysis.py        -> a reusable analysis script / Jupyter-friendly module
  2) cord19_streamlit_app.py   -> a simple Streamlit app that depends only on the metadata.csv file

Instructions
------------
1. Download `metadata.csv` from the CORD-19 dataset and place it in the same folder as these scripts.
   (If the file is very large, you can work with a subset using `--nrows` or sample options in `cord19_analysis.py`.)

2. Install required libraries:
   pip install pandas numpy matplotlib wordcloud streamlit

3. To run the analysis script from the command line (quick demo with sampling):
   python cord19_analysis.py --demo

4. To run the Streamlit app (recommended, interactive):
   Save the second part into `cord19_streamlit_app.py` (or copy it to a file) and run:
   streamlit run cord19_streamlit_app.py

------

### FILE: cord19_analysis.py

# Save from here into a file named `cord19_analysis.py`

"""
# cord19_analysis.py

import os
import argparse
from collections import Counter
from turtle import st
from typing import Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import streamlit as st
from collections import Counter


def load_metadata(path: str, usecols: Optional[List[str]] = None, nrows: Optional[int] = None,
                  sample_frac: Optional[float] = None, chunksize: Optional[int] = None) -> pd.DataFrame:
    """Load metadata.csv with several memory-friendly options.

    - usecols: list of column names to read (recommended if file is huge)
    - nrows: read only first nrows (quick tests)
    - sample_frac: if provided, read full file in chunks and sample approx sample_frac rows
    - chunksize: used with sampling (if sample_frac provided); default 100k
    """
    if sample_frac is not None:
        if chunksize is None:
            chunksize = 100_000
        reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize)
        sampled_parts = []
        for chunk in reader:
            frac = sample_frac
            # sample from chunk (may lead to slightly non-uniform sample but works well)
            sampled_parts.append(chunk.sample(frac=frac))
        df = pd.concat(sampled_parts, ignore_index=True)
        return df
    else:
        df = pd.read_csv(path, usecols=usecols, nrows=nrows)
        return df


def basic_exploration(df: pd.DataFrame, head: int = 5) -> None:
    """Print basic information about the DataFrame and return some common objects."""
    print("\n--- HEAD ---")
    print(df.head(head))
    print("\n--- SHAPE ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\n--- DTYPE ---")
    print(df.dtypes)
    print("\n--- MISSING VALUES (top columns) ---")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing.head(20))
    print("\n--- NUMERICAL SUMMARY ---")
    if df.select_dtypes(include=np.number).shape[1] > 0:
        print(df.describe())
    else:
        print("No numeric columns to summarize.")


def clean_metadata(df: pd.DataFrame, drop_threshold: float = 0.7) -> pd.DataFrame:
    """Clean the metadata DataFrame in-place and return it.

    - drop columns with > drop_threshold missing values
    - parse publish_time -> datetime and extract year
    - create title_word_count and abstract_word_count
    """
    df = df.copy()

    # Drop columns with too many missing values
    missing_frac = df.isnull().mean()
    cols_to_drop = missing_frac[missing_frac > drop_threshold].index.tolist()
    if cols_to_drop:
        print(f"Dropping columns with > {drop_threshold*100:.0f}% missing: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)

    # Standard publish time parsing (CORD-19 uses 'publish_time' in many versions)
    if 'publish_time' in df.columns:
        df['publish_time_parsed'] = pd.to_datetime(df['publish_time'], errors='coerce')
    else:
        # try common alternatives
        for alt in ['publish_date', 'date']:
            if alt in df.columns:
                df['publish_time_parsed'] = pd.to_datetime(df[alt], errors='coerce')
                break
        else:
            df['publish_time_parsed'] = pd.NaT

    df['year'] = df['publish_time_parsed'].dt.year

    # Word counts
    for col in ['title', 'abstract']:
        if col in df.columns:
            df[f'{col}_word_count'] = df[col].fillna("").astype(str).map(lambda x: len(x.split()))

    return df


def count_papers_by_year(df: pd.DataFrame) -> pd.Series:
    if 'year' not in df.columns:
        raise ValueError("DataFrame must contain 'year' column. Run clean_metadata first.")
    year_counts = df['year'].value_counts(dropna=True).sort_index()
    return year_counts


def top_journals(df: pd.DataFrame, topn: int = 20) -> pd.Series:
    # common column names: 'journal', 'journal_name'
    journal_col = None
    for col in ['journal', 'journal_name', 'venue']:
        if col in df.columns:
            journal_col = col
            break
    if journal_col is None:
        raise ValueError('No journal-like column found (looked for journal, journal_name, venue)')
    return df[journal_col].value_counts().head(topn)


def title_word_frequency(df: pd.DataFrame, topn: int = 50, stopwords: Optional[set] = None) -> List[tuple]:
    if 'title' not in df.columns:
        raise ValueError("No 'title' column found")
    if stopwords is None:
        stopwords = set(STOPWORDS)
    titles = df['title'].dropna().astype(str).str.lower()
    words = Counter()
    for t in titles:
        for w in (''.join(ch if ch.isalnum() else ' ' for ch in t)).split():
            if w and w not in stopwords and len(w) > 2:
                words[w] += 1
    return words.most_common(topn)


# Simple plotting helpers

def plot_publications_over_time(year_counts: pd.Series, savepath: Optional[str] = None) -> None:
    plt.figure(figsize=(10, 5))
    year_counts = year_counts.dropna()
    plt.bar(year_counts.index.astype(int), year_counts.values)
    plt.xlabel('Year')
    plt.ylabel('Number of publications')
    plt.title('Publications by Year')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        print(f"Saved plot to {savepath}")
    else:
        plt.show()


def plot_top_journals(journal_series: pd.Series, savepath: Optional[str] = None) -> None:
    plt.figure(figsize=(10, 6))
    journal_series.plot(kind='bar')
    plt.xlabel('Journal')
    plt.ylabel('Paper count')
    plt.title('Top Journals')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        print(f"Saved plot to {savepath}")
    else:
        plt.show()


def generate_wordcloud_from_titles(df: pd.DataFrame, max_words: int = 200, savepath: Optional[str] = None):
    text = ' '.join(df['title'].dropna().astype(str).tolist())
    stopwords = set(STOPWORDS)
    wc = WordCloud(width=1200, height=600, max_words=max_words, stopwords=stopwords)
    img = wc.generate(text)
    plt.figure(figsize=(20, 10))
    plt.imshow(img, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        print(f"Saved wordcloud to {savepath}")
    else:
        plt.show()
    return img


def get_source_distribution(df: pd.DataFrame) -> pd.Series:
    # column 'source_x' is used by many CORD-19 releases
    col = None
    for c in ['source_x', 'source', 'dataset']:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError('No source-like column found (looked for source_x, source, dataset)')
    return df[col].value_counts()


def save_sample_csv(df: pd.DataFrame, outpath: str, n: int = 1000):
    df.sample(n=min(n, len(df))).to_csv(outpath, index=False)
    print(f"Saved sample ({min(n,len(df))} rows) to {outpath}")


# Example CLI: demo run
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CORD-19 metadata quick analysis')
    parser.add_argument('--metadata', type=str, default='metadata.csv', help='Path to metadata.csv')
    parser.add_argument('--nrows', type=int, default=None, help='Read only first N rows')
    parser.add_argument('--sample_frac', type=float, default=None, help='Read with sampling fraction')
    parser.add_argument('--demo', action='store_true', help='Run a short demo (load small sample, produce plots)')
    args = parser.parse_args()

    if args.demo:
        print('Running demo: reading up to 20000 rows or sampling from file if large...')
        # Prefer sampling if file big
        if args.nrows is None and args.sample_frac is None:
            # Try to read with nrows first to be safe
            try:
                df = load_metadata(args.metadata, usecols=['title', 'abstract', 'publish_time', 'journal', 'source_x'], nrows=20000)
            except Exception as e:
                print('Failed to read first 20000 rows directly, trying smaller or sampling approach:', e)
                df = load_metadata(args.metadata, usecols=['title', 'abstract', 'publish_time', 'journal', 'source_x'], nrows=5000)
        else:
            df = load_metadata(args.metadata, usecols=['title', 'abstract', 'publish_time', 'journal', 'source_x'], nrows=args.nrows, sample_frac=args.sample_frac)

        basic_exploration(df)
        df_clean = clean_metadata(df)
        yc = count_papers_by_year(df_clean)
        print('\nYear counts:\n', yc)
        plot_publications_over_time(yc)
        try:
            tj = top_journals(df_clean, topn=10)
            print('\nTop journals:\n', tj)
            plot_top_journals(tj)
        except Exception as e:
            print('Could not compute top_journals:', e)
        wc = generate_wordcloud_from_titles(df_clean, max_words=100)
        print('Demo complete.')

#end of cord19_analysis.py

