import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

class PlagiarismChecker:
    def __init__(self, min_similarity=0.7):
        self.min_similarity = min_similarity

    def search_web(self, query, num_results=5, google_api_key=None, google_cx=None):
        """
        Use Google Custom Search API. Requires API key and CX (search engine ID).
        Returns a list of URLs for the query.
        """
        import os
        # Hardcoded API key for demo (not secure for production)
        api_key = google_api_key or "AIzaSyCmvJjUyS_LyU2zqT6rdvSFsgQpaB_68Pg"
        cx = google_cx or os.environ.get('GOOGLE_CSE_ID')
        if not api_key or not cx:
            print("Google API key or CSE ID not set. Set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables or pass as arguments.")
            return []
        params = {
            "key": api_key,
            "cx": cx,
            "q": query,
            "num": num_results
        }
        try:
            resp = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
            data = resp.json()
            results = []
            for item in data.get("items", []):
                link = item.get("link")
                if link:
                    results.append(link)
                if len(results) >= num_results:
                    break
            return results
        except Exception as e:
            print(f"Google Custom Search error: {e}")
            return []

    def fetch_text_from_url(self, url):
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            return ' '.join([p.get_text() for p in paragraphs])
        except Exception:
            return ""

    def check_plagiarism(self, input_text):
        sources = []
        search_results = self.search_web(input_text)
        for url in search_results:
            page_text = self.fetch_text_from_url(url)
            similarity = SequenceMatcher(None, input_text, page_text).ratio()
            if similarity >= self.min_similarity:
                sources.append({'url': url, 'similarity': similarity})
        return sources

    def generate_report(self, input_text):
        sources = self.check_plagiarism(input_text)
        report = {
            'input_text': input_text,
            'sources': sources,
            'plagiarized': bool(sources)
        }
        return report
