import requests
from io import BytesIO
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.probability import FreqDist
from collections import Counter
import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import STOPWORDS

# Download the Basel Framework document
url = "https://www.bis.org/baselframework/BaselFramework.pdf"
response = requests.get(url)
f = BytesIO(response.content)

# Extract the text from the PDF
reader = PdfReader(f)
text = ''
for page in reader.pages:
    text += page.extract_text()

# Preprocessing
# Tokenize the text
tokens = word_tokenize(text)
print(tokens[:100])

# Part-of-speech tagging
pos_tags = pos_tag(tokens)

# Named Entity Recognition
namedEnt = ne_chunk(pos_tags)

keywords = ['probability', 'default', 'loss', 'downturn', 'discounting', 'cashflow', 'data', 'credit', 'risk', 'requirements']

# Get POS tags for keywords
keywords_pos_tags = nltk.pos_tag(keywords)

# Print POS tags
for word, pos in keywords_pos_tags:
    print(f"{word}: {pos}")

# Print named entities that are in the keywords list
print([chunk for chunk in namedEnt if hasattr(chunk, 'label') and chunk.label() == 'ORGANIZATION' and any(leaf[0] in keywords for leaf in chunk.leaves())])

# Create a Text object for concordance
text_obj = nltk.Text(tokens)

# Print concordance for each keyword
for keyword in keywords:
    print(f"Occurrences of {keyword}:")
    text_obj.concordance(keyword)

# Word Cloud
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=800, height=400, stopwords=stopwords).generate(' '.join(tokens))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Frequency Distribution Plot
fdist = FreqDist(tokens)
plt.figure(figsize=(10, 5))
fdist.plot(30)  # Plot the 30 most common tokens

# Bar Chart of Keyword Frequency
keyword_counts = Counter([token.lower() for token in tokens if token.lower() in keywords])
plt.figure(figsize=(10, 5))
plt.bar(keyword_counts.keys(), keyword_counts.values())
plt.title('Keyword Frequency')
plt.xlabel('Keywords')
plt.ylabel('Frequency')
plt.show()


# TF-IDF Analysis
documents = [text]
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
df_tfidf = pd.DataFrame(tfidf_matrix.T.todense(), index=tfidf_vectorizer.get_feature_names_out(), columns=["tfidf"])
df_tfidf = df_tfidf.sort_values(by=["tfidf"], ascending=False)
print(df_tfidf.head(10))

# Sentiment Analysis (using TextBlob)
from textblob import TextBlob

blob = TextBlob(text)
sentiment = blob.sentiment
print(f"Sentiment Analysis:\nPolarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")
