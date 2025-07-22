import re
import string
import contractions
import nltk
import os
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Tentukan path lokal untuk nltk_data
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.mkdir(nltk_data_path)

# Tambahkan path lokal ke nltk
nltk.data.path.append(nltk_data_path)

# Download resource NLTK jika belum tersedia
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path)
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_path)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path)
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_path)

# Siapkan stopwords dan lemmatizer
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
  if treebank_tag.startswith('J'):
    return wordnet.ADJ
  elif treebank_tag.startswith('V'):
    return wordnet.VERB
  elif treebank_tag.startswith('N'):
    return wordnet.NOUN
  elif treebank_tag.startswith('R'):
    return wordnet.ADV
  else:
    return wordnet.NOUN

def preprocess_text(text):
    # 1. Expand contractions (e.g., can't → cannot)
    text = contractions.fix(text)
    
    # 2. Lowercase
    text = text.lower()

    # 3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 4. Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 5. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    text = " ".join([word for word in str(text).split() if word not in stopwords])

    # 7. Lemmatization
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    lemmatized = [wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged_tokens]
    cleaned_tokens = ' '.join(lemmatized)

    return cleaned_tokens