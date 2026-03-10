from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tempfile, os
from pdf_parser import PDFParser
from arxiv_search import ArXivSearch
from llm_client import LocalLLMClient, MockLLMClient, CloudLLMClient, GeminiClient
from agents import run_agents

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.post("/api/review")
async def review_paper(
    file: UploadFile = File(...),
    llm_backend: str = Form("local"),   # "local" | "mock" | "cloud" | "gemini"
):
    """
    llm_backend options:
      local   — Ollama llama3.2:3b (requires Ollama running locally)
      mock    — instant deterministic responses, for UI testing
      claude  — Anthropic Claude Haiku (requires ANTHROPIC_API_KEY env var)
      gemini  — Google Gemini 1.5 Flash (requires GEMINI_API_KEY env var, free tier)
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        print(f"\n{'='*60}")
        print(f"[1/4] Extracting PDF: {file.filename}")
        print(f"{'='*60}")
        parser = PDFParser()
        text = parser.extract_text_from_pdf(tmp_path)
        sections = parser.parse_sections(text)
        print(f"      ✓ {len(text)} chars extracted")
        for k, v in sections.items():
            if k != "full_text":
                print(f"        - {k}: {len(v)} chars")

        print(f"\n{'='*60}")
        print(f"[2/4] Searching arXiv")
        print(f"{'='*60}")
        arxiv_summaries = ArXivSearch().search_arxiv(sections.get("title", ""))
        print(f"      ✓ {arxiv_summaries.count('Title:')} papers found")

        print(f"\n{'='*60}")
        print(f"[3/4] Initializing LLM backend: {llm_backend}")
        print(f"{'='*60}")
        if llm_backend == "mock":
            llm_client = MockLLMClient()
            print("      ✓ Mock LLM (instant, for testing)")
        elif llm_backend == "claude":
            llm_client = CloudLLMClient()
            print(f"      ✓ Claude ({llm_client.model}) — parallel mode")
        elif llm_backend == "gemini":
            llm_client = GeminiClient()
            print(f"      ✓ Gemini ({llm_client.model}) — parallel mode, 1M ctx")
        else:
            llm_client = LocalLLMClient()
            print(
                f"      ✓ Local Ollama ({llm_client.model}) — parallel + RAG "
                f"(workers={llm_client.max_parallel_requests}, timeout={llm_client.timeout}s)"
            )

        print(f"\n{'='*60}")
        print(f"[4/4] Running multi-agent review (RAG enabled)")
        print(f"{'='*60}")
        review = await run_agents(sections, arxiv_summaries, llm_client)

        print(f"\n{'='*60}")
        print(f"✓ REVIEW COMPLETE")
        print(f"{'='*60}\n")
        return review

    except Exception as exc:
        print(f"\n✗ ERROR: {exc}\n")
        return {"error": str(exc)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)