from flask import Flask, request, render_template
import pickle
import numpy as np
from scipy.spatial import distance

app = Flask(__name__)

#Load GloVe model
def load_GloVe(f="glove_model.pkl"):
    with open(f, "rb") as file:
        model = pickle.load(file)
        
    return model["embeddings"], model["word2index"]

glove_embeddings, word2index = load_GloVe("glove_model.pkl")


#Load corpus
with open('tokenized_corpus.pkl', "rb") as file:
    corpus = pickle.load(file)

flattened_corpus = [word for sentence in corpus for word in sentence]
corpus = list(set(flattened_corpus))  #Unique words only

from scipy.spatial import distance

#Similarity function
def cos_sim(a, b):                           
    return 1 - distance.cosine(a, b)

#Function to look up top 10 similar words with query
def search_similar(query, corpus, glove_embeddings, top_n=10):
    if query not in glove_embeddings:
        return f"Query '{query}' not found in embeddings."
    
    query_embedding = glove_embeddings[query]

    similarities = []
    for context in corpus:
        if context in glove_embeddings:
            if context != query and context in glove_embeddings:
                context_embedding = glove_embeddings[context]
                similarity = cos_sim(query_embedding, context_embedding)
                similarities.append((context, similarity))

    
    results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    return "", results



#main
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error = ""
    if request.method == "POST":
        query = request.form["query"].strip().lower()
        error, results = search_similar(query, corpus, glove_embeddings, top_n=10)
    return render_template("index.html", results=results, error=error)


if __name__ == "__main__":
    app.run(debug=True)