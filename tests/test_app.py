import pytest
from backend.pdf_parser import PDFParser
from backend.arxiv_search import ArXivSearch
from backend.llm_client import MockLLMClient
from backend.agents import MethodologyReviewer, NoveltyReviewer, ClarityReviewer, EvidenceReviewer, MetaReviewer, run_agents
import asyncio


def test_parse_sections():
    parser = PDFParser()
    text = """
    Sample Paper Title

    Abstract
    This is the abstract.

    Methodology
    This is the methodology.

    Results
    These are the results.

    Conclusion
    This is the conclusion.
    """
    sections = parser.parse_sections(text)
    assert "Sample Paper Title" in sections["title"]
    assert "abstract" in sections["abstract"]
    assert "methodology" in sections["methodology"]


def test_arxiv_search():
    # Mock or skip real API call
    searcher = ArXivSearch()
    summaries = searcher.search_arxiv("Test Title")
    assert isinstance(summaries, str)


@pytest.mark.asyncio
async def test_agents():
    llm = MockLLMClient()
    sections = {
        "methodology": "Test methodology",
        "abstract": "Test abstract",
        "conclusion": "Test conclusion",
        "full_text": "Full text",
        "results": "Test results"
    }
    arxiv = "Mock arxiv summaries"

    review = await run_agents(sections, arxiv, llm)
    assert "strengths" in review
    assert "scores" in review
    assert "final_score" in review["scores"]