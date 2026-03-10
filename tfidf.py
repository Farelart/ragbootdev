import json
import string
import os
import pickle
import math
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter

with open('data/stopwords.txt') as f:
    stop_words_obj = [w.strip() for w in f.readlines()]

with open('data/movies.json', 'r') as f:
    movies_obj = json.load(f)

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
     self.index = defaultdict(set)
     self.docmap = dict()
     self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id, text):
        tokens = preprocessed_text(text)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        term=term.lower()
        term=stemmer.stem(term)
        return sorted(self.index.get(term, set()))
    
    def get_tf(self, doc_id, term):
        tokens = preprocessed_text(term)
        if len(tokens) > 1:
            raise ValueError(f"Expected a single token, got {len(tokens)}: {tokens}")
        if len(tokens) == 0:
            return 0
        token = tokens[0]
        return self.term_frequencies[doc_id].get(token, 0)
    
    def get_idf(self, term):
        term = term.lower()
        term = stemmer.stem(term)
        df = len(self.index.get(term, set()))
        return math.log((len(self.docmap) + 1) / (df + 1))
    
    def get_tfidf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf


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

        with open('cache/tf_path.pkl', 'wb') as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        if not os.path.exists('cache/index.pkl') or \
           not os.path.exists('cache/docmap.pkl') or \
           not os.path.exists('cache/tf_path.pkl'):
            raise FileNotFoundError("Cache files not found. Please build the index first.")
        with open('cache/index.pkl', 'rb') as f:
            self.index = pickle.load(f)
        with open('cache/docmap.pkl', 'rb') as f:
            self.docmap = pickle.load(f)
        with open('cache/tf_path.pkl', 'rb') as f:
            self.term_frequencies = pickle.load(f)

def preprocessed_text(text : str) -> list[str]:
    text=text.lower()
    text=text.translate(str.maketrans("", "", string.punctuation))
    tokens=text.split()
    tokens=[t for t in tokens if t not in stop_words_obj]
    tokens=[stemmer.stem(t) for t in tokens]
    return tokens

def main():
    idx = InvertedIndex()
    if not os.path.exists('cache/index.pkl') and not os.path.exists('cache/docmap.pkl') and not os.path.exists('cache/tf_path.pkl'):
        print("Building the index")
        idx.build()
        idx.save()
        print("Index built and saved.")
    else:
        try:
            idx.load()
            print("Index loaded from cache.")
        except FileNotFoundError as e:
            print(e)
            return
    
    query = input('Enter a keyword : ')
    query_tokens = preprocessed_text(query)

    results = set()
    for token in query_tokens:
        docs_token = idx.get_documents(token)
        for doc_id in docs_token:
            results.add(doc_id)
            if len(results) >= 5:
                break
        if len(results) >= 5:
            break

    for doc_id in sorted(results):
        print(f"ID {doc_id} | Title : {idx.docmap[doc_id]['title']}")

    doc_id_term = int(input("In which document id you want to search in : "))
    term = input("Which term : ")
    print(f"tf is : {idx.get_tf(doc_id_term, term)}")
    print(f"idf is : {idx.get_idf(term)}")
    print(f"tf-idf is : {idx.get_tfidf(doc_id_term, term)}")

if __name__ == "__main__":
    main()