import requests
from typing import List, Dict


class ArXivSearch:
    def __init__(self):
        pass

    def search_arxiv(self, title: str) -> str:
        """Search arXiv for related papers based on title and return summaries."""
        # Generate search queries
        queries = [
            f'ti:"{title}"',
            f'all:"{title}"',
            f'abs:"{title}"'
        ]

        summaries = []
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
                        summaries.append(f"Title: {title_elem.text}\nAbstract: {summary_elem.text[:500]}...")

        return "\n\n".join(summaries[:5])  # Limit to 5 summaries