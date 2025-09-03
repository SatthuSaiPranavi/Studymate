import fitz  # PyMuPDF

def extract_text_from_pdf(file_stream):
    """Extract readable text from a PDF file-like object."""
    if isinstance(file_stream, str):
        doc = fitz.open(file_stream)
    else:
        data = file_stream.read()
        doc = fitz.open(stream=data, filetype='pdf')

    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    doc.close()
    return "\n".join(texts)
