from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex, Document
import pandas as pd


# Similarity top k
k = 3
# Temperature
temp = 0.1
# Maximum tokens num_output
mt = 1024

with open("documents.txt", "r") as d:
    document_content = d.read()
documents = [Document(text=document_content)]
print(documents)

vector_store_index = VectorStoreIndex.from_documents(documents)
vector_query_engine = vector_store_index.as_query_engine(
    similarity_top_k=k,
    temperature=temp,
    num_output=mt
)


model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity(text1, text2):
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    similarity = cosine_similarity([embeddings1], [embeddings2])
    return similarity[0][0]

def index_query(input_query):
    response = vector_query_engine.query(input_query)
    print(str(response))
    node_data = []
    for node_with_score in response.source_nodes:
        node = node_with_score.node
        node_info = {
            'Node ID': node.id_,
            'Score': node_with_score.score,
            'Text': node.text
        }
        node_data.append(node_info)
    df = pd.DataFrame(node_data)
    return df, response