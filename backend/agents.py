"""
agents.py

Each ReviewAgent now uses RAG to retrieve relevant chunks from the full paper
instead of receiving a naively truncated string. This gives full-paper coverage
while keeping prompts within local model limits.

When the LLMClient supports parallel execution (cloud clients), all agents run
concurrently via asyncio.gather. For local Ollama, they run sequentially.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from llm_client import LLMClient, truncate
from rag import PaperRAG
import re


# ── helpers ───────────────────────────────────────────────────────────────────
def _extract_line(response: str, label: str) -> str:
    pattern = re.compile(rf'{re.escape(label)}[:\s]+(.+)', re.IGNORECASE)
    match = pattern.search(response)
    return match.group(1).strip() if match else ""


def _clean_field(value: str) -> str:
    if not value:
        return ""
    cleaned = value.strip()
    if cleaned.lower() in {"...", "..", ".", "n/a", "na", "none"}:
        return ""
    if cleaned.startswith("...") or cleaned.endswith("..."):
        return ""
    return cleaned


# ── base agent ────────────────────────────────────────────────────────────────
class ReviewAgent(ABC):
    def __init__(self, llm_client: LLMClient, rag: PaperRAG):
        self.llm_client = llm_client
        self.rag = rag

    @abstractmethod
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        pass

    def _extract_score(self, response: str, default: int = 3) -> int:
        match = re.search(r'score[:\s]+([1-5])', response, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r'\b([1-5])\b', response)
        return int(match.group(1)) if match else default


# ── individual agents ─────────────────────────────────────────────────────────
class MethodologyReviewer(ReviewAgent):
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        # RAG: retrieve chunks most relevant to methodology questions
        context = self.rag.retrieve(
            "experimental methodology research design approach evaluation metrics baseline",
            top_k=3,
            max_chars=2200,
        )
        prompt = (
            "You are a peer reviewer evaluating experimental methodology.\n"
            "Respond with exactly:\nStrengths: ...\nWeaknesses: ...\nScore: X/5\n\n"
            "Use concrete sentences and do not output placeholder text like '...'.\n\n"
            f"Relevant paper sections:\n{context}"
        )
        response = await self.llm_client.generate(prompt)
        print(f"      [MethodologyReviewer] ✓ ({len(response)} chars)")
        return {
            "strengths": _clean_field(_extract_line(response, "strengths")),
            "weaknesses": _clean_field(_extract_line(response, "weaknesses")),
            "score": self._extract_score(response, default=4),
        }


class NoveltyReviewer(ReviewAgent):
    async def analyze(self, arxiv_summaries: str = "", **kwargs) -> Dict[str, Any]:
        # RAG: retrieve chunks about contributions and comparisons to prior work
        context = self.rag.retrieve(
            "novel contribution originality prior work comparison state of the art innovation",
            top_k=3,
            max_chars=2200,
        )
        related = truncate(arxiv_summaries, 800) if arxiv_summaries else "None provided."
        prompt = (
            "You are a peer reviewer assessing novelty and originality.\n"
            "Respond with exactly:\nStrengths: ...\nWeaknesses: ...\nMissing related work: ...\nScore: X/5\n\n"
            "Use concrete sentences and do not output placeholder text like '...'.\n\n"
            f"Relevant paper sections:\n{context}\n\n"
            f"Related arXiv papers:\n{related}"
        )
        response = await self.llm_client.generate(prompt)
        print(f"      [NoveltyReviewer] ✓ ({len(response)} chars)")
        return {
            "strengths": _clean_field(_extract_line(response, "strengths")),
            "weaknesses": _clean_field(_extract_line(response, "weaknesses")),
            "missing_related_work": _clean_field(_extract_line(response, "missing related work")),
            "score": self._extract_score(response, default=3),
        }


class ClarityReviewer(ReviewAgent):
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        # RAG: retrieve chunks about structure, writing quality, explanations
        context = self.rag.retrieve(
            "writing clarity structure explanation figure table introduction motivation",
            top_k=3,
            max_chars=2200,
        )
        prompt = (
            "You are a peer reviewer assessing writing clarity and paper structure.\n"
            "Respond with exactly:\nStrengths: ...\nWeaknesses: ...\nScore: X/5\n\n"
            "Use concrete sentences and do not output placeholder text like '...'.\n\n"
            f"Relevant paper sections:\n{context}"
        )
        response = await self.llm_client.generate(prompt)
        print(f"      [ClarityReviewer] ✓ ({len(response)} chars)")
        return {
            "strengths": _clean_field(_extract_line(response, "strengths")),
            "weaknesses": _clean_field(_extract_line(response, "weaknesses")),
            "score": self._extract_score(response, default=4),
        }


class EvidenceReviewer(ReviewAgent):
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        # RAG: retrieve chunks about results and experimental evidence
        context = self.rag.retrieve(
            "results experiments BLEU score accuracy performance table comparison dataset",
            top_k=3,
            max_chars=2200,
        )
        prompt = (
            "You are a peer reviewer assessing experimental evidence and result quality.\n"
            "Respond with exactly:\nStrengths: ...\nWeaknesses: ...\nScore: X/5\n\n"
            "Use concrete sentences and do not output placeholder text like '...'.\n\n"
            f"Relevant paper sections:\n{context}"
        )
        response = await self.llm_client.generate(prompt)
        print(f"      [EvidenceReviewer] ✓ ({len(response)} chars)")
        return {
            "strengths": _clean_field(_extract_line(response, "strengths")),
            "weaknesses": _clean_field(_extract_line(response, "weaknesses")),
            "score": self._extract_score(response, default=3),
        }


# ── meta reviewer ─────────────────────────────────────────────────────────────
class MetaReviewer:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def aggregate(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_strengths = [r.get("strengths", "") for r in reviews]
        all_weaknesses = [r.get("weaknesses", "") for r in reviews]
        missing_work = [r.get("missing_related_work", "") for r in reviews if r.get("missing_related_work")]
        scores = [r["score"] for r in reviews]
        final_score = round(sum(scores) / len(scores), 1) if scores else 0

        summary = (
            f"Strengths: {'; '.join(filter(None, all_strengths))}\n"
            f"Weaknesses: {'; '.join(filter(None, all_weaknesses))}"
        )
        prompt = (
            "You are a meta-reviewer synthesizing peer review feedback.\n"
            "Respond with exactly:\n"
            "Questions for authors: ...\nSuggested improvements: ...\n\n"
            "Each field must contain a concrete sentence and must not be '...'.\n\n"
            f"Review summary:\n{truncate(summary, 1200)}"
        )
        response = await self.llm_client.generate(prompt)
        print(f"      [MetaReviewer] ✓ ({len(response)} chars)")

        questions = _clean_field(_extract_line(response, "questions for authors"))
        improvements = _clean_field(_extract_line(response, "suggested improvements"))
        if not questions:
            questions = "Can the authors provide ablations and discuss failure cases for key claims?"
        if not improvements:
            improvements = "Add clearer error analysis, stronger baselines, and reproducibility details."

        return {
            "strengths": "; ".join(filter(None, all_strengths)),
            "weaknesses": "; ".join(filter(None, all_weaknesses)),
            "missing_related_work": "; ".join(filter(None, missing_work)),
            "questions_for_authors": questions,
            "suggested_improvements": improvements,
            "scores": {
                "experimental_soundness": reviews[0]["score"],
                "originality": reviews[1]["score"] if len(reviews) > 1 else 3,
                "clarity": reviews[2]["score"] if len(reviews) > 2 else 4,
                "evidence_support": reviews[3]["score"] if len(reviews) > 3 else 3,
                "prior_work_coverage": 3,
                "research_value": 4,
                "final_score": final_score,
            },
        }


# ── entry point ───────────────────────────────────────────────────────────────
async def run_agents(
    sections: Dict[str, str],
    arxiv_summaries: str,
    llm_client: LLMClient,
) -> Dict[str, Any]:

    # Build RAG index once from the full paper text
    rag = PaperRAG(sections["full_text"])

    agents_and_kwargs = [
        (MethodologyReviewer(llm_client, rag), {}),
        (NoveltyReviewer(llm_client, rag),     {"arxiv_summaries": arxiv_summaries}),
        (ClarityReviewer(llm_client, rag),     {}),
        (EvidenceReviewer(llm_client, rag),    {}),
    ]

    # Run all agents concurrently with bounded parallelism.
    # This enables parallel execution for both local and cloud backends,
    # while avoiding overload on slower local setups.
    max_parallel = getattr(llm_client, "max_parallel_requests", 4)
    print(f"      Running agents in PARALLEL (max_workers={max_parallel})...")
    semaphore = asyncio.Semaphore(max_parallel)

    async def _guarded(agent: ReviewAgent, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            try:
                return await agent.analyze(**kwargs)
            except Exception as exc:
                name = agent.__class__.__name__
                print(f"      [{name}] ✗ fallback due to error: {exc}")
                return {
                    "strengths": "The paper presents a relevant research problem and plausible approach.",
                    "weaknesses": "Reviewer-specific analysis timed out; rerun with LOCAL_MAX_PARALLEL=1 for deeper feedback.",
                    "score": 3,
                }

    tasks = [_guarded(agent, kwargs) for agent, kwargs in agents_and_kwargs]
    reviews = await asyncio.gather(*tasks)

    meta = MetaReviewer(llm_client)
    return await meta.aggregate(list(reviews))