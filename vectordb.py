import json
import re
import torch
import torch.nn.functional as F
from train_embeddings import SentenceEncoder, Vocab, tokenize, EMBED_DIM, HIDDEN_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_vocab():
    with open("vocab.json", "r") as f:
        data = json.load(f)
    word2idx = data["word2idx"]
    max_len = data["max_len"]

    vocab = Vocab.__new__(Vocab)  
    vocab.word2idx = word2idx
    vocab.idx2word = {i: w for w, i in word2idx.items()}

    model = SentenceEncoder(len(vocab), EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load("sentence_encoder.pt", map_location=DEVICE))
    model.eval()

    return model, vocab, max_len



class TinyVectorDB:
    def __init__(self):
        self.vectors = []     
        self.documents = []   

    def add(self, embedding, document):
        self.vectors.append(embedding)
        self.documents.append(document)

    def search(self, query_embedding, k=3):
        if not self.vectors:
            return []
        all_vecs = torch.stack(self.vectors)        
        query = query_embedding.unsqueeze(0)          

        all_vecs_norm = F.normalize(all_vecs, dim=1)
        query_norm = F.normalize(query, dim=1)

        sims = (all_vecs_norm @ query_norm.T).squeeze(1)  # (N,)
        top_k = torch.topk(sims, min(k, len(self.documents)))

        results = []
        for score, idx in zip(top_k.values.tolist(), top_k.indices.tolist()):
            results.append((self.documents[idx], score))
        return results



def split_into_sentences(paragraph):

    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    return [s.strip() for s in sentences if s.strip()]


def embed_sentence(model, vocab, max_len, sentence):
    encoded = vocab.encode(sentence, max_len)
    tensor = torch.tensor([encoded], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        embedding = model(tensor).squeeze(0)  
    return embedding


if __name__ == "__main__":
    model, vocab, max_len = load_model_and_vocab()


    paragraph = """
    The kitten was sleeping on the soft couch. The playful puppy was running across the green lawn. The experienced physician successfully treated the patient. The shiny automobile was parked near the house. Heavy rain covered the road during the afternoon. A gentle breeze moved through the trees before the downpour began. The teacher explained the lesson with patience and clarity. The student carefully read an interesting book in the library. The baker prepared fresh bread and warm pastry for breakfast. The family welcomed a rescue feline into their home. The loyal dog guarded the residence throughout the night. The mechanic repaired the damaged vehicle before the long journey. The joyful doctor smiled after the successful operation. A fluffy cat rested on the comfortable chair. The powerful laptop handled the new software efficiently. The little kitten chased a playful puppy across the yard. Everyone enjoyed the enthusiastic barking of the excited dog. The smartphone battery lasted throughout the entire day. The surgeon explained the procedure with empathy and patience. The quiet home became brighter with a happy kitten running around.
    """

    sentences = split_into_sentences(paragraph)
    print(f"Found {len(sentences)} sentences in the paragraph\n")

    db = TinyVectorDB()
    for s in sentences:
        emb = embed_sentence(model, vocab, max_len, s)
        db.add(emb, s)

    print("--- Vector DB built ---\n")


    test_queries = [
        "A small cat was playing.",
        "The vehicle needed repairs.",
        "I felt very happy today.",
    ]

    for query in test_queries:
        query_emb = embed_sentence(model, vocab, max_len, query)
        results = db.search(query_emb, k=3)
        print(f"Query: {query!r}")
        for doc, score in results:
            print(f"    [{score:.3f}] {doc}")
        print()
