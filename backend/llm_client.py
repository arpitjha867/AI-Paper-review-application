import ollama
from typing import Dict, Any
import anthropic
"""
llm_client.py

Available clients:
  LocalLLMClient   — Ollama local model (llama3.2:3b). Sequential, slower.
  ClaudeClient     — Anthropic Claude API. Fast, large context, parallel-safe.
                     pip install anthropic  |  export ANTHROPIC_API_KEY=sk-ant-...
  GeminiClient     — Google Gemini API. Free tier, 1M token context window.
                     pip install google-generativeai  |  export GEMINI_API_KEY=AIza...
  MockLLMClient    — Instant deterministic responses for testing.
"""

import asyncio
import os
from abc import ABC, abstractmethod

# ── constants ────────────────────────────────────────────────────────────────
LLM_TIMEOUT_SECONDS = 180
MAX_PROMPT_CHARS = 3500  # Fallback when RAG is not active


def truncate(text: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


# ── base ─────────────────────────────────────────────────────────────────────
class LLMClient(ABC):
    # Cloud clients set this True — enables parallel asyncio.gather in agents
    supports_parallel: bool = False
    context_limit_chars: int = MAX_PROMPT_CHARS
    max_parallel_requests: int = 4

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass


# ── local ollama ──────────────────────────────────────────────────────────────
class LocalLLMClient(LLMClient):
    supports_parallel = True
    context_limit_chars = 2800
    max_parallel_requests = 2

    def __init__(self, model: str = "llama3.2:3b", timeout: int = LLM_TIMEOUT_SECONDS):
        self.model = model
        self.timeout = timeout
        # Allow runtime tuning without code changes.
        self.max_parallel_requests = int(os.getenv("LOCAL_MAX_PARALLEL", "2"))
        self.max_retries = int(os.getenv("LOCAL_LLM_MAX_RETRIES", "2"))

    async def generate(self, prompt: str) -> str:
        working_prompt = truncate(prompt, self.context_limit_chars)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        ollama.generate,
                        model=self.model,
                        prompt=working_prompt,
                        options={"temperature": 0.2, "num_predict": 320},
                    ),
                    timeout=self.timeout,
                )
                if isinstance(response, dict):
                    return response.get("response", "")
                return getattr(response, "response", "") or ""
            except asyncio.TimeoutError:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Local LLM timed out after {self.timeout}s (retries={self.max_retries}). "
                        "Try: set LOCAL_MAX_PARALLEL=1, use mock/cloud backend, or pull a faster model."
                    )
                working_prompt = truncate(working_prompt, 1800)
                await asyncio.sleep(0.4 * attempt)
            except Exception as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Local LLM failed: {exc}")
                await asyncio.sleep(0.4 * attempt)


from ollama import chat
from ollama import ChatResponse

class CloudLLMClient(LLMClient):
    """Client for Ollama Cloud models (requires Ollama account and cloud access)"""
    def __init__(self, model: str = "deepseek-v3.2:cloud"):
        self.model = model
        self.api_key = "6480a5f72c384d348476bfa55ba7c3cd.nBnaUkyK3RCUCUKeOsOQSBUR"

    async def generate(self, prompt: str) -> str:
        try:
            response: ChatResponse = chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            # For cloud models, you may need different configuration
            # Ollama cloud models use the same API but may require authentication

            print(response)
            if self.api_key:
                # If API key is set, you might need to configure headers or different endpoint
                import ollama
                # Note: ollama library may need to be configured for cloud
                response = ollama.generate(model=self.model, prompt=prompt)
            else:
                response = ollama.generate(model=self.model, prompt=prompt)
            return response.get('response', '')
        except Exception as exc:
            raise RuntimeError(f"Cloud LLM generation failed: {exc}. Make sure you have Ollama Cloud access and the model is available.")




# ── anthropic claude ──────────────────────────────────────────────────────────
class ClaudeClient(LLMClient):
    """
    Fast, high-quality, parallel-safe. Best for production use.
    Models:
      claude-3-haiku-20240307     — cheapest, still very capable
      claude-3-5-sonnet-20241022  — best quality
    Setup:
      pip install anthropic
      export ANTHROPIC_API_KEY=sk-ant-...
    """
    supports_parallel = True
    context_limit_chars = 150_000   # ~100k tokens
    max_parallel_requests = 6

    def __init__(self, model: str = "claude-3-haiku-20240307"):
        self.model = model
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable.")

    async def generate(self, prompt: str) -> str:
        try:

            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            message = await client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as exc:
            raise RuntimeError(f"Claude API failed: {exc}")


# ── google gemini ─────────────────────────────────────────────────────────────
class GeminiClient(LLMClient):
    """
    Free tier: 15 req/min on gemini-1.5-flash with a 1M token context window.
    Best option if you want cloud speed at zero cost.
    Setup:
      pip install google-generativeai
      export GEMINI_API_KEY=AIza...
    Get key: https://aistudio.google.com/app/apikey
    """
    supports_parallel = True
    context_limit_chars = 500_000   # Gemini 1.5 flash: ~750k tokens
    max_parallel_requests = 6

    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set the GEMINI_API_KEY environment variable.")

    async def generate(self, prompt: str) -> str:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            # SDK is sync — run in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: model.generate_content(prompt)
            )
            return response.text
        except Exception as exc:
            raise RuntimeError(f"Gemini API failed: {exc}")


# ── mock ──────────────────────────────────────────────────────────────────────
class MockLLMClient(LLMClient):
    """Zero-latency deterministic client for testing the pipeline."""
    supports_parallel = True
    context_limit_chars = 999_999
    max_parallel_requests = 8

    async def generate(self, prompt: str) -> str:
        await asyncio.sleep(0.05)
        p = prompt.lower()
        if "methodology" in p:
            return "Strengths: Solid experimental design with clear baselines.\nWeaknesses: Assumptions not clearly stated.\nScore: 4/5"
        elif "novelty" in p:
            return "Strengths: Novel attention mechanism with strong motivation.\nWeaknesses: Overlaps with prior work.\nMissing related work: Recent transformer variants.\nScore: 3/5"
        elif "clarity" in p:
            return "Strengths: Clear writing throughout.\nWeaknesses: Some sections verbose.\nScore: 4/5"
        elif "evidence" in p:
            return "Strengths: Claims supported by BLEU scores.\nWeaknesses: Limited language diversity.\nScore: 3/5"
        else:
            return "Questions for authors: How does this scale to longer sequences?\nSuggested improvements: Add ablation studies.\nScore: 3/5"