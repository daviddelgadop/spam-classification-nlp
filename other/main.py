import re
import unicodedata
import base64
import emoji
import nltk
#from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tag import PerceptronTagger

import spacy
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # avoid randomness in langdetect

# Load spaCy English and French models
nlp_en = spacy.load("en_core_web_md")
nlp_fr = spacy.load("fr_core_news_md")

nltk.download("punkt")
nltk.download("stopwords")
nltk.download('omw-1.4')
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

# ------------------------------------------------------
# CLEANING FUNCTIONS
# ------------------------------------------------------
def remove_mime(text):
    text = re.sub(r"-{2,}[_A-Za-z0-9=]+(?:NextPart|Part|boundary)[A-Za-z0-9_=.-]*", " ", text, flags=re.I)
    text = re.sub(r"^(Content|MIME|Multipart|Boundary|Return|Delivered|Message-ID|X-[A-Za-z\-]+)[^\n]*",
                  " ", text, flags=re.I | re.M)
    return text


def demojize_words(text):
    text = emoji.demojize(text)
    text = re.sub(r":([a-zA-Z0-9_]+):", r" :\1: ", text)
    text = re.sub(r":([a-zA-Z0-9_]+):", lambda m: m.group(1).replace("_", " "), text)
    return text


def clean_text(text):

    text = demojize_words(text)

    text = re.sub(r"<[^<>]*?>", " ", text)

    # URL before MIME removal
    text = re.sub(r"https?://\S+", " <url> ", text)
    text = re.sub(r"/[A-Za-z_][A-Za-z0-9_\-/]*(\?[A-Za-z0-9_\-=&]+)?", " <url> ", text)

    # Base64 decode
    def try_decode_b64(s):
        try:
            return base64.b64decode(s).decode("utf-8")
        except:
            return s
    text = re.sub(r"\b[A-Za-z0-9+/=]{8,}\b", lambda m: try_decode_b64(m.group(0)), text)

    # ⭐ Remove fragments of broken URLs (tracking garbage)
    #text = re.sub(r"\b[A-Za-z0-9]{25,}\b", " ", text)
    # text = re.sub(r"\b[A-Za-z0-9]{12,}\b", " ", text)
    text = re.sub(r"\b(?=.*[A-Za-z])(?=.*[0-9])[A-Za-z0-9]{6,}\b", " ", text)

    # ⭐ NOW remove MIME
    text = remove_mime(text)

    #print(text)


    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Collapse sigle
    def collapse_sigle(match):
        return match.group(0).replace(".", "")
    text = re.sub(r"\b(?:[A-Za-z]\.){2,}", collapse_sigle, text)

    # Emails
    def email_to_tokens(match):
        email = match.group(0)
        for ch in [".", "_", "@", "-"]:
            email = email.replace(ch, " ")
        return email
    EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    text = re.sub(EMAIL_REGEX, email_to_tokens, text)

    text = text.replace("@", "a")

    for inv in ["\u00A0", "\u2007", "\u202F", "\uFEFF"]:
        text = text.replace(inv, " ")

    text = text.replace("\\n", " ")
    text = re.sub(r"[^A-Za-z. ]", " ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)

    return text.lower().strip()




# ------------------------------------------------------
# POS mapping for English WordNet
# ------------------------------------------------------

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


# ------------------------------------------------------
# MULTILINGUAL LEMMATIZATION
# ------------------------------------------------------

def lemmatize_multilingual(words):
    lemmatizer_en = WordNetLemmatizer()
    tagger = PerceptronTagger(lang="eng")
    tagged_en = tagger.tag(words)

    final_lemmas = []

    for i, word in enumerate(words):
        try:
            lang = detect(word)
        except:
            lang = "en"  # fallback

        if lang == "fr":
            # spaCy FR
            doc = nlp_fr(word)
            lemma = doc[0].lemma_
            final_lemmas.append(lemma)

        else:
            # English fallback
            tag = tagged_en[i][1]
            wn_pos = get_wordnet_pos(tag)
            lemma = lemmatizer_en.lemmatize(word, wn_pos)
            final_lemmas.append(lemma)

    return final_lemmas


# ------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------

with open("text_to_clean.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

cleaned = clean_text(raw_text)
print("\nClean text:\n", cleaned)

words = word_tokenize(cleaned)

# Stopwords FR+EN
stop_words = set(stopwords.words("english")) | set(stopwords.words("french"))
filtered_words = [w for w in words if w.casefold() not in stop_words]

print("\nFiltered words:\n", filtered_words)

# Stemming English only
stemmer_en = SnowballStemmer("english")
stemmed_words = [stemmer_en.stem(w) for w in filtered_words]

# Multilingual lemmatization
lemmatized_words = lemmatize_multilingual(filtered_words)

print("\nStemmed words:\n", stemmed_words)
print("\nLemmatized words:\n", lemmatized_words)



