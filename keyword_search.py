import json
import string
from nltk.stem import PorterStemmer

with open('data/stopwords.txt', 'r') as f:
        stop_words_obj = [w.strip() for w in f.readlines()]

with open('data/movies.json', 'r') as f:
        movies_obj = json.load(f)

stemmer = PorterStemmer()

def main():
    query = input("Enter a keyword: ")
    movies = movies_obj['movies']

    i = 0
    for movie in movies:
        query_tokens = preprocessed_text(query)
        title_tokens = preprocessed_text(movie['title'])
        if has_matching_token(query_tokens, title_tokens):
            i += 1
            print(f"{i}. {movie['title']}")
    if i == 0:
        print("no match found")
        
def preprocessed_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words_obj]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
            
    return False

if __name__ == "__main__":
    main()
