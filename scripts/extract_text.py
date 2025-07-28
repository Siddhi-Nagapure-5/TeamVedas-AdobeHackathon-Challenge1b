import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import tempfile
from multiprocessing import Pool
import re

# 🧠 Configure tesseract path (Windows users)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🛠️ Optional flag to include page separators
ENABLE_PAGE_HEADERS = False

def normalize_text(text):
    """
    Cleans up raw extracted text.
    - Normalize bullets
    - Remove excessive newlines
    - Strip whitespace
    """
    if not text:
        return ""

    # Normalize bullets: replace o, -, ● with •
    text = re.sub(r"^[\-\*•o●]+\s*", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)  # Reduce >2 newlines to 2
    return text.strip()

def extract_text_from_pdf(pdf_path):
    all_text = ""
    scanned_pages = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 10:
                    text = normalize_text(text)
                    if ENABLE_PAGE_HEADERS:
                        all_text += f"\n\n--- Page {i+1} ---\n"
                    all_text += text + "\n"
                else:
                    scanned_pages.append(i)

        if scanned_pages:
            images = convert_from_path(pdf_path)
            for i in scanned_pages:
                with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
                    images[i].save(tmp.name, "PNG")
                    ocr_text = pytesseract.image_to_string(Image.open(tmp.name))
                    ocr_text = normalize_text(ocr_text)
                    if ENABLE_PAGE_HEADERS:
                        all_text += f"\n\n--- OCR Page {i+1} ---\n"
                    all_text += ocr_text + "\n"

    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")

    return all_text.strip()

def process_single_pdf(args):
    pdf_path, output_path = args
    print(f"🔍 Processing: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Saved: {output_path}")

def gather_tasks_for_collection(collection_number):
    base_dir = os.getcwd()
    input_pdf_dir = os.path.join(base_dir, f"Collection {collection_number}", "PDFs")
    output_txt_dir = os.path.join(base_dir, f"Collection {collection_number} Extracted", "PDFs")

    tasks = []
    if os.path.exists(input_pdf_dir):
        for filename in os.listdir(input_pdf_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(input_pdf_dir, filename)
                output_path = os.path.join(output_txt_dir, filename.replace(".pdf", ".txt"))
                tasks.append((pdf_path, output_path))
    return tasks

def process_all_collections():
    all_tasks = []
    for i in range(1, 4):  # For Collection 1, 2, 3
        tasks = gather_tasks_for_collection(i)
        all_tasks.extend(tasks)

    with Pool(processes=8) as pool:
        pool.map(process_single_pdf, all_tasks)

if __name__ == "__main__":
    process_all_collections()
    print("\n🎉 All collections processed and extracted in parallel.")
