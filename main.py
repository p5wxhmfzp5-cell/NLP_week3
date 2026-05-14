from config import CONFIG
from data import load_triviaqa
from retrievers import BM25Retriever
from generators import Generator
from evaluate import compute_em

# Open a file for writing. 'w' mode will create the file or overwrite it.
output_file = open("C:/Users/giehp/OneDrive - JGU/Uni/NLP Applications in Research and Industry/NLP_week3/output_week3.txt", 'w')

def run_experiment(dataset, retriever, generator, k):
    predictions = []
    golds = []
    

    for sample in dataset:
        print(type(dataset))
        print(dataset[:3])
        print(type(dataset[0]))
        print(dataset[0])

        q = sample("question")
        gold = sample["answer"]["aliases"]

        if k == 0:
            docs = []
        else:
            docs = retriever.retrieve(q, k)

        pred = generator.generate(q, docs)

        predictions.append(pred)
        golds.append(gold)

    return compute_em(predictions, golds)


def main():
    dataset = load_triviaqa()
   
    corpus = [x["question"] for x in dataset]  # placeholder corpus
    
    bm25 = BM25Retriever(corpus)
    generator = Generator(CONFIG["generators"]["flan"])

    for k in CONFIG["k_values"]:
        score = run_experiment(dataset[:100], bm25, generator, k)
        print(f"K={k}, EM={score:.3f}")

if __name__ == "__main__":
    main()