import sys
import subprocess
try:
    from pypdf import PdfReader
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pypdf'])
    from pypdf import PdfReader

try:
    reader = PdfReader(r'C:\Users\arjun\Downloads\LLM_Assignment.pdf')
    links = []
    for page in reader.pages:
        if '/Annots' in page:
            for annot in page['/Annots']:
                annot_obj = annot.get_object()
                if '/A' in annot_obj and '/URI' in annot_obj['/A']:
                    links.append(annot_obj['/A']['/URI'])
    print("Found links:")
    for link in links:
        print(link)
except Exception as e:
    print(f'Error reading PDF: {e}')
