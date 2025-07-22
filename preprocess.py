import re
import string
import contractions
import nltk
import os
from nltk.corpus import stopwords
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

# Siapkan stopwords dan lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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

    # 6. Remove stopwords & 7. Lemmatization
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(cleaned_tokens)