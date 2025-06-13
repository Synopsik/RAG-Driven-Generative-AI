import anthropic
import os
import time
from dotenv import load_dotenv

from Chapter01.RAG_Naive import calculate_cosine_similarity
from Chapter01.RAG_Overview_1 import calculate_enhanced_similarity

load_dotenv()
from RAG_Util import call_llm
from RAG_Naive import find_best_match_keyword_search
from records import db_records

def main():
    start_time = time.time()
    #client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    query = "Define a rag store."

    best_keyword_record, best_matching_record = find_best_match_keyword_search(query, db_records)
    score = calculate_cosine_similarity(query, best_matching_record)
    similarity_score = calculate_enhanced_similarity(query, best_matching_record)


    print(f"Best Keyword Score: {best_keyword_record}\n"
          f"Best Matching Record: {best_matching_record}\n"
          f"Best Cosine Similarity Score: {score:.3f}\n"
          f"Best Enhanced Similarity Score: {similarity_score:.3f}")

    #response = call_llm(client, query)
    stop_time = time.time()
    finish_time = stop_time - start_time
    print(f"Time to complete: {finish_time:.2f}\n"
          f"response")

if __name__ == "__main__":
    main()