import re
import random
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


DATA_FILE = "synthetic_embedding_corpus.txt"  
EMBED_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-3
TEMPERATURE = 0.1
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")



def load_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    # remove stray curly braces like {delight} seen in a couple of lines
    lines = [line.replace("{", "").replace("}", "") for line in lines]
    return lines

def tokenize(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
    return sentence.split()


def cluster_by_template(sentences, min_overlap_ratio=0.75):
    """
    Groups sentences that are likely synonym-swapped versions of each other.
    Two sentences are in the same cluster if they have the same word count
    and share at least `min_overlap_ratio` of words at the same position.
    """
    tokenized = [tokenize(s) for s in sentences]
    n = len(sentences)
    parent = list(range(n))  # union-find

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    by_length = defaultdict(list)
    for i, toks in enumerate(tokenized):
        by_length[len(toks)].append(i)

    for length, idxs in by_length.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                toks_i, toks_j = tokenized[i], tokenized[j]
                if length == 0:
                    continue
                matches = sum(1 for x, y in zip(toks_i, toks_j) if x == y)
                if matches / length >= min_overlap_ratio:
                    union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)


    return [idxs for idxs in clusters.values() if len(idxs) >= 2]


class Vocab:
    def __init__(self, sentences):
        vocab_set = set()
        for s in sentences:
            vocab_set.update(tokenize(s))
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        for w in sorted(vocab_set):
            self.word2idx[w] = len(self.word2idx)
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, sentence, max_len=20):
        toks = tokenize(sentence)[:max_len]
        ids = [self.word2idx.get(t, 1) for t in toks]
        ids = ids + [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.project = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, token_ids):

        mask = (token_ids != 0).unsqueeze(-1).float()  
        embedded = self.embedding(token_ids)        
        output, _ = self.lstm(embedded)               
        
        summed = (output * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        return self.project(pooled)                     
        
        
def contrastive_loss(anchor, positive, temperature=TEMPERATURE):
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    logits = anchor @ positive.T / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2



def build_pairs(sentences, clusters):
    pairs = []
    for idxs in clusters:

        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pairs.append((sentences[idxs[a]], sentences[idxs[b]]))
    return pairs


def train():
    sentences = load_sentences(DATA_FILE)
    print(f"Loaded {len(sentences)} sentences")

    clusters = cluster_by_template(sentences)
    print(f"Found {len(clusters)} template clusters")
    total_pairs = sum(len(c) * (len(c) - 1) // 2 for c in clusters)
    print(f"Total positive pairs available: {total_pairs}")

    if total_pairs < BATCH_SIZE:
        raise ValueError(
            f"Not enough pairs ({total_pairs}) for batch size {BATCH_SIZE}. "
            f"Lower BATCH_SIZE or check clustering."
        )

    vocab = Vocab(sentences)
    print(f"Vocab size: {len(vocab)}")

    pairs = build_pairs(sentences, clusters)
    random.shuffle(pairs)

    model = SentenceEncoder(len(vocab)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    max_len = max(len(tokenize(s)) for s in sentences)
    print(f"Max sentence length (tokens): {max_len}")

    for epoch in range(1, EPOCHS + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(pairs) - BATCH_SIZE + 1, BATCH_SIZE):
            batch = pairs[i:i + BATCH_SIZE]
            anchors = [vocab.encode(p[0], max_len) for p in batch]
            positives = [vocab.encode(p[1], max_len) for p in batch]

            anchor_tensor = torch.tensor(anchors, dtype=torch.long, device=DEVICE)
            positive_tensor = torch.tensor(positives, dtype=torch.long, device=DEVICE)

            optimizer.zero_grad()
            anchor_embed = model(anchor_tensor)
            positive_embed = model(positive_tensor)

            loss = contrastive_loss(anchor_embed, positive_embed)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:3d}/{EPOCHS} | avg loss: {avg_loss:.4f}")


    torch.save(model.state_dict(), "sentence_encoder.pt")
    with open("vocab.json", "w") as f:
        json.dump({"word2idx": vocab.word2idx, "max_len": max_len}, f)

    print("\nTraining complete. Saved sentence_encoder.pt and vocab.json")
    return model, vocab, max_len


def quick_test(model, vocab, max_len):
    model.eval()
    test_sentences = [
        "The cat sat on the mat.",
        "The kitten sat on the mat.",
        "The truck drove down the highway.",
        "I felt great joy today.",
        "I felt great happiness today.",
    ]

    with torch.no_grad():
        encoded = torch.tensor(
            [vocab.encode(s, max_len) for s in test_sentences],
            dtype=torch.long, device=DEVICE
        )
        embeddings = model(encoded)
        embeddings = F.normalize(embeddings, dim=1)

    print("\n--- Quick similarity check ---")
    for i in range(len(test_sentences)):
        for j in range(i + 1, len(test_sentences)):
            sim = (embeddings[i] @ embeddings[j]).item()
            print(f"sim({test_sentences[i]!r}, {test_sentences[j]!r}) = {sim:.3f}")


if __name__ == "__main__":
    model, vocab, max_len = train()
    quick_test(model, vocab, max_len)
