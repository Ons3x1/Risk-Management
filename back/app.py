from flask import Flask, request, jsonify
import joblib
import numpy as np
from combinedGraphModel import CombinedGraphModel
# Load the trained model
combined_model = joblib.load("models/combined_model.pkl")
G = joblib.load("models/G.pkl")
data = joblib.load("models/data.pkl")

app = Flask(__name__)

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import nn

# Load the pre-trained all-mpnet-base-v2 model and tokenizer from Hugging Face
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
mpnet_model = AutoModel.from_pretrained(model_name)

# Define the function for query-based node inference
def find_similar_nodes(query, k, G, data, combined_model):
    result = []

    # Step 1: Embed the query using the mpnet model
    def embed_query_mpnet(query):
        # Tokenize and create input tensor for mpnet model
        inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = mpnet_model(**inputs)

        # Use the last hidden state to get the embeddings (average over all tokens)
        query_embedding = outputs.last_hidden_state.mean(dim=1)
        return query_embedding

    # Embed the query
    query_embedding = embed_query_mpnet(query)
    
    # Step 2: Reduce the dimensionality of the query embedding from 768 to 8 using a Linear layer
    target_dim = 8  # Target dimension to match node embeddings
    linear_layer = nn.Linear(query_embedding.shape[1], target_dim)

    # Apply the linear layer to reduce the dimensionality
    query_embedding_reduced = linear_layer(query_embedding)

    # Normalize the reduced embedding to unit length
    query_embedding_reduced = F.normalize(query_embedding_reduced, p=2, dim=1)
    print(data)
    print("test")
    # Step 3: Extract node embeddings from the trained model (assuming `combined_model` is already trained)
    combined_model.eval()
    
    with torch.no_grad():
        node_embeddings, _, _, _ = combined_model(data)
    print("test")
    # Step 4: Normalize node embeddings for cosine similarity
    node_embeddings = F.normalize(node_embeddings, p=2, dim=1)
    print("test")
    # Step 5: Calculate cosine similarity between query and each node
    cos_similarities = F.cosine_similarity(query_embedding_reduced, node_embeddings)

    # Step 6: Find the `k` most similar nodes
    top_k_indices = torch.topk(cos_similarities, k=k).indices

    # Step 7: Create a mapping between NetworkX node IDs and PyTorch Geometric node indices
    node_mapping = {idx: node_id for idx, node_id in enumerate(G.nodes())}

    # Step 8: Print the most similar nodes, their similarity scores, and their attributes from the NetworkX graph `G`
    print(f"\nTop {k} nodes most similar to the query '{query}':")
    for i in top_k_indices:
        node_idx = i.item()
        original_node_id = node_mapping[node_idx]  # Get the original NetworkX node ID
        similarity_score = cos_similarities[node_idx].item()

        # Extract attributes of the node from the original NetworkX graph `G`
        node_attributes = G.nodes[original_node_id]
        result.append(node_attributes)
        # Display the similarity score and the attributes of the node
        print(f"Node: {original_node_id} (Index {node_idx}): Similarity Score = {similarity_score:.4f}")
        print(f"Node: {original_node_id} Attributes: {node_attributes}\n")
    return(result)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the POST request
        print("---------------------------")
        print("step 1 request")
        d = request.get_json(force=True)
        print("---------------------------")
        print(d)
        k=d["key"]
        print("---------------------------")
        print(k)
        query=d["query"]
        print("---------------------------")
        print(query)
        # Process data (assumes data is a list of features)
        r=find_similar_nodes(query, k, G, data, combined_model)
        print("---------------------------")
        print(r)

        # Return prediction as JSON
        return jsonify({"prediction": r})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
