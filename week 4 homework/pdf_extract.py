import requests
import xml.etree.ElementTree as ET # XML parsing library
import time
from pathlib import Path

def get_arxiv_papers(max_results=10):
    url = "http://export.arxiv.org/api/query"
    params = {
        'search_query': 'cat:cs.CL',
        'start': 0,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    
    response = requests.get(url, params=params)
    root = ET.fromstring(response.text)
    
    # Parse the XML - need to handle namespaces
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    
    papers = []
    for entry in root.findall('atom:entry', namespace):
        title = entry.find('atom:title', namespace).text.strip()
        paper_id = entry.find('atom:id', namespace).text.split('/')[-1]
        
        # Find PDF link
        pdf_url = None
        for link in entry.findall('atom:link', namespace):
            if link.get('title') == 'pdf':
                pdf_url = link.get('href')
                break
        
        papers.append({
            'id': paper_id,
            'title': title,
            'pdf_url': pdf_url
        })
    
    return papers

def download_pdfs(papers, download_dir="pdfs"):
    """Download PDFs and add local_pdf_path to each paper dict"""
    Path(download_dir).mkdir(exist_ok=True)
    
    for i, paper in enumerate(papers):
        if not paper['pdf_url']:
            continue
            
        filepath = Path(download_dir) / f"{paper['id'].replace('/', '_')}.pdf"
        
        if filepath.exists():
            print(f"Already have {filepath.name}")
        else:
            print(f"Downloading {i+1}/{len(papers)}: {paper['id']}")
            response = requests.get(paper['pdf_url'])
            filepath.write_bytes(response.content)
            time.sleep(1)  # Be nice to arXiv
        
        paper['local_pdf_path'] = str(filepath)
    
    return papers

# Usage:
papers = get_arxiv_papers(50)
papers = download_pdfs(papers)
