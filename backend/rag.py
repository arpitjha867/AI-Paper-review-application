from typing import List
import re, math
from collections import Counter


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


class PaperRAG:
    def __init__(self, text: str, chunk_size: int = 400, overlap: int = 80):
        self.chunks = self._chunk(text, chunk_size, overlap)
        self.tokenized_chunks = [_tokenize(c) for c in self.chunks]
        self.idf = self._build_idf(self.tokenized_chunks)
        print(f"      [RAG] Built index: {len(self.chunks)} chunks from {len(text)} chars")

    def _chunk(self, text: str, size: int, overlap: int) -> List[str]:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunks.append(" ".join(words[i: i + size]))
            i += size - overlap
        return chunks

    def _build_idf(self, tokenized_chunks: List[List[str]]) -> dict:
        N = len(tokenized_chunks)
        df: dict = {}
        for tokens in tokenized_chunks:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        return {t: math.log(N / (1 + df[t])) for t in df}

    def _tfidf_score(self, query_tokens: List[str], chunk_tokens: List[str]) -> float:
        chunk_counter = Counter(chunk_tokens)
        total = len(chunk_tokens) or 1
        score = 0.0
        for token in set(query_tokens):
            tf = chunk_counter.get(token, 0) / total
            score += tf * self.idf.get(token, 0)
        return score

    def retrieve(self, query: str, top_k: int = 5, max_chars: int = 3500) -> str:
        query_tokens = _tokenize(query)
        scored = [
            (self._tfidf_score(query_tokens, self.tokenized_chunks[i]), i)
            for i in range(len(self.chunks))
        ]
        scored.sort(reverse=True)
        top_chunks = [self.chunks[i] for _, i in scored[:top_k]]
        combined = "\n\n---\n\n".join(top_chunks)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n...[truncated]"
        return combined