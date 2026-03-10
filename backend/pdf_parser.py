import fitz  # PyMuPDF
import re
from typing import Dict


class PDFParser:
    def __init__(self):
        pass

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract full text from a PDF file."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def parse_sections(self, text: str) -> Dict[str, str]:
        """Parse the extracted text into sections using regex heuristics.
        
        Handles academic papers with structure:
        - Title (before Abstract, authors, affiliations)
        - Abstract
        - Introduction
        - Methodology/Methods/Approach
        - Results/Experiments/Findings
        - Conclusion/Discussion/Summary
        """
        sections = {
            "title": "",
            "abstract": "",
            "introduction": "",
            "methodology": "",
            "results": "",
            "conclusion": "",
            "full_text": text
        }

        # Extract title - look for text before Abstract, excluding author emails and affiliations
        title_match = re.search(
            r'^([A-Za-z\s\-\(\)]+?)(?:\n\s*(?:[A-Za-z\s,\.]+(?:@[^\n]+)?|Abstract))',
            text,
            re.MULTILINE
        )
        if title_match:
            title_text = title_match.group(1).strip()
            # Clean up title - remove extra whitespace and common prefixes
            title_text = re.sub(r'\s+', ' ', title_text)
            if len(title_text) > 10 and len(title_text) < 300:  # Reasonable title length
                sections["title"] = title_text

        # Extract abstract - more flexible pattern
        abstract_match = re.search(
            r'(?:^|\n)\s*(?:ABSTRACT|Abstract)\s*\n(.*?)(?=\n\s*(?:1\s|Introduction|1\.|INTRODUCTION|Methodology|Methods|2\s))',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if abstract_match:
            sections["abstract"] = abstract_match.group(1).strip()

        # Extract introduction
        intro_match = re.search(
            r'(?:^|\n)\s*(?:1\s+)?(?:INTRODUCTION|Introduction)\s*\n(.*?)(?=\n\s*(?:2\s|METHOD|METHODOLOGY|Methodology|Approach|RELATED WORK))',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if intro_match:
            sections["introduction"] = intro_match.group(1).strip()

        # Extract methodology/methods
        methodology_match = re.search(
            r'(?:^|\n)\s*(?:2\s+)?(?:METHODOLOGY|Methodology|METHOD|Methods|METHOD|Approach|APPROACH)\s*\n(.*?)(?=\n\s*(?:3\s|RESULTS|Results|EXPERIMENTS|Experiments|EVALUATION|4\s))',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if methodology_match:
            sections["methodology"] = methodology_match.group(1).strip()

        # Extract results
        results_match = re.search(
            r'(?:^|\n)\s*(?:3\s+)?(?:RESULTS|Results|EXPERIMENTS|Experiments|FINDINGS|Findings|EVALUATION|Evaluation)\s*\n(.*?)(?=\n\s*(?:4\s|DISCUSSION|Discussion|CONCLUSION|Conclusion|5\s))',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if results_match:
            sections["results"] = results_match.group(1).strip()

        # Extract conclusion
        conclusion_match = re.search(
            r'(?:^|\n)\s*(?:4\s+)?(?:CONCLUSION|Conclusion|SUMMARY|Summary|DISCUSSION|Discussion|FUTURE WORK)\s*\n(.*?)(?=\n\s*(?:5\s|REFERENCES|References|ACKNOWLEDGMENTS|Acknowledgments|$))',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if conclusion_match:
            sections["conclusion"] = conclusion_match.group(1).strip()

        return sections