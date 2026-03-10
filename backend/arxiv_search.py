import re
from difflib import SequenceMatcher
import requests


class ArXivSearch:
    def __init__(self):
        pass

    def _normalize_title(self, value: str) -> str:
        """Lowercase and remove punctuation/extra spaces for robust comparison."""
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", (value or "").lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _is_same_paper(self, input_title: str, candidate_title: str) -> bool:
        """
        Detect if arXiv candidate title is the same as uploaded paper title.
        Uses exact normalized match + fuzzy similarity fallback.
        """
        a = self._normalize_title(input_title)
        b = self._normalize_title(candidate_title)

        if not a or not b:
            return False

        if a == b:
            return True

        # Handles minor punctuation/version/case variations.
        return SequenceMatcher(None, a, b).ratio() >= 0.92

    def search_arxiv(self, title: str) -> str:
        """Search arXiv for related papers based on title and return summaries."""
        # Generate search queries
        queries = [
            f'ti:"{title}"',
            f'all:"{title}"',
            f'abs:"{title}"'
        ]

        summaries = []
        seen_titles = set()
        for query in queries[:2]:  # Limit to 2 queries to avoid overload
            url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=3"
            response = requests.get(url)
            if response.status_code == 200:
                # Parse XML response (simplified)
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                    summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                    if title_elem is not None and summary_elem is not None:
                        candidate_title = (title_elem.text or "").strip()

                        # Exclude the same paper from related-work grounding.
                        if self._is_same_paper(title, candidate_title):
                            continue

                        normalized = self._normalize_title(candidate_title)
                        if normalized in seen_titles:
                            continue

                        seen_titles.add(normalized)
                        summaries.append(
                            f"Title: {candidate_title}\n"
                            f"Abstract: {(summary_elem.text or '')[:500]}..."
                        )

        return "\n\n".join(summaries[:5])  # Limit to 5 summaries