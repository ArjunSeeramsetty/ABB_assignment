import sys
import subprocess
try:
    from pypdf import PdfReader
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pypdf'])
    from pypdf import PdfReader

try:
    reader = PdfReader(r'C:\Users\arjun\Downloads\LLM_Assignment.pdf')
    text = '\n'.join(page.extract_text() for page in reader.pages)
    with open('pdf_text_utf8.txt', 'w', encoding='utf-8') as f:
        f.write(text)
except Exception as e:
    print(f'Error reading PDF: {e}')
