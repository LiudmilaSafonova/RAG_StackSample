import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
translator = str.maketrans('', '', string.punctuation)


def preprocess(text):
    text = text.lower()    # lower case
    text = re.sub(r'\d+', '', text)  # markdown code
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # markdown links
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)    # no digits
    text = re.sub(r"<.*?>", "", text)   # no html tags
    text = text.translate(translator)  # punctuation
    word_tokens = text.split()
    filtered_text = [word for word in word_tokens if word not in stop_words]
    lemmas = [lemmatizer.lemmatize(word) for word in filtered_text]
    return ' '.join(lemmas)

