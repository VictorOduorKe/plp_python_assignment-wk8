# cord19_streamlit_app.py

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud, STOPWORDS

from  status import load_metadata, clean_metadata, title_word_frequency

# Load data
@st.cache_data
def load_and_clean_data(path="metadata.csv", nrows=5000):
    df = load_metadata(path, nrows=nrows)
    df = clean_metadata(df)
    return df

df = load_and_clean_data()

# For now, just use df directly (later you can add filters)
filtered = df.copy()

@st.cache_data
def show_filtered_records_info(filtered: pd.DataFrame, df: pd.DataFrame):
    st.write(f"Filtered records: {filtered.shape[0]} out of {df.shape[0]} total records.")
    if filtered.shape[0] == 0:
        st.warning("No records match the filter criteria.")
    else:
        st.success(f"{filtered.shape[0]} records match the filter criteria.")

# Layout
col1, col2 = st.columns([2, 1])
show_filtered_records_info(filtered, df)

with col1:
    st.subheader('Publications by year')
    if 'year' in filtered.columns and filtered['year'].notna().any():
        year_counts = filtered['year'].dropna().astype(int).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(year_counts.index, year_counts.values)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of papers')
        st.pyplot(fig)
    else:
        st.write("No valid 'year' data to plot.")

    st.subheader('Top journals (filtered)')
    if 'journal' in filtered.columns and filtered['journal'].notna().any():
        topj = filtered['journal'].value_counts().head(20)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        topj.plot(kind='bar', ax=ax2)
        ax2.set_ylabel('Count')
        ax2.set_xlabel('Journal')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig2)
    else:
        st.write('No journal column available in your metadata slice.')

with col2:
    st.subheader('Word cloud of titles (filtered)')
    text = ' '.join(filtered['title'].dropna().astype(str).tolist()) if 'title' in filtered.columns else ''
    if text:
        wc = WordCloud(width=800, height=400, stopwords=set(STOPWORDS)).generate(text)
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.imshow(wc, interpolation='bilinear')
        ax3.axis('off')
        st.pyplot(fig3)
    else:
        st.write('No title text available for word cloud.')

st.header('Data sample (first 200 rows)')
st.dataframe(filtered.head(200))

st.header('Most frequent words in titles')
words = title_word_frequency(filtered, topn=50)
if words:
    df_words = pd.DataFrame(words, columns=['word', 'count'])
    st.table(df_words.head(30))
else:
    st.write('No words found.')

