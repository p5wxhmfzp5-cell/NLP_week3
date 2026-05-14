CONFIG = {
    "datasets": [
        "mandarjoshi/trivia_qa",
        "sentence-transformers/natural-questions"
    ],
    
    "k_values": [0, 1, 5, 10],
    
    "generators": {
        "flan": "google/flan-t5-large",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2"
    },
    
    "retrievers": {
        "bm25": "bm25",
        "dpr": {
            "q_encoder": "facebook/dpr-question_encoder-multiset-base",
            "ctx_encoder": "facebook/dpr-ctx_encoder-multiset-base"
        },
        "contriever": "facebook/contriever"
    },
    
    "max_context_length": 2048,
    "top_k_docs": 10,
}