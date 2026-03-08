import json
import string
import os
import pickle
from nltk.stem import PorterStemmer


with open('data/stopwords.txt') as f:
    stop_words_obj = [w.strip() for w in f.readlines()]

with open('data/movies.json', 'r') as f:
    movies_obj = json.load(f)

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
     self.index = dict()
     self.docmap = dict()

    def __add_document(self, doc_id, text):
        tokens = preprocessed_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)


    def get_documents(self, term):
        term=term.lower()
        term=stemmer.stem(term)
        return sorted(self.index.get(term, set()))

    def build(self):
        movies=movies_obj['movies']
        for doc_id, movie in enumerate(movies):
            self.docmap[doc_id] = movie
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

    def save(self):
        os.makedirs('cache', exist_ok=True)
        
        with open('cache/index.pkl', 'wb') as f:
            pickle.dump(self.index, f)

        with open('cache/docmap.pkl', 'wb') as f:
            pickle.dump(self.docmap, f)



def preprocessed_text(text : str) -> list[str]:
    text=text.lower()
    text=text.translate(str.maketrans("", "", string.punctuation))
    tokens=text.split()
    tokens=[t for t in tokens if t not in stop_words_obj]
    tokens=[stemmer.stem(t) for t in tokens]
    return tokens

def main():
    idx = InvertedIndex()
    if not os.path.exists('cache/index.pkl') and not os.path.exists('cache/docmap.pkl'):
        idx.build()
        idx.save()
        print("Index built and saved.")
    else:
        with open('cache/index.pkl', 'rb') as f:
            idx.index = pickle.load(f)
        with open('cache/docmap.pkl', 'rb') as f:
            idx.docmap = pickle.load(f)
        print("Index loaded from cache.")
    
    query = input('Enter a keyword :')
    print(idx.get_documents(query))

if __name__ == "__main__":
    main()