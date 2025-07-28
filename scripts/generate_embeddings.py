import os
import json
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool
import datetime
import re

# ✅ Global model variable
model = None

def init_worker():
    global model
    model = SentenceTransformer('./models/minilm')  # Load local model once per process

def is_heading(line):
    """
    Heuristic: short lines with capital letters or ending in ':' or numbered.
    """
    if len(line.strip()) > 100:
        return False
    if line.strip().endswith(":"):
        return True
    if line.isupper():
        return True
    if re.match(r'^[A-Z][\w\s\-]{0,40}$', line):
        return True
    if re.match(r'^\d+[\.\)]\s+[A-Z]', line):  # e.g. 1. Introduction
        return True
    return False

def chunk_by_headings(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    sections = []
    current_heading = None
    current_content = []

    for line in lines:
        if is_heading(line):
            if current_heading and current_content:
                sections.append({
                    "heading": current_heading,
                    "content": " ".join(current_content)
                })
            current_heading = line
            current_content = []
        else:
            current_content.append(line)

    # Last block
    if current_heading and current_content:
        sections.append({
            "heading": current_heading,
            "content": " ".join(current_content)
        })

    return sections

def fallback_chunk(text, max_words=120):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk_words = words[i:i+max_words]
        content = " ".join(chunk_words)
        heading = chunk_words[0] if chunk_words else "Untitled"
        chunks.append({
            "heading": heading,
            "content": content
        })
    return chunks

def process_file(args):
    collection_num, input_dir, filename = args
    file_path = os.path.join(input_dir, filename)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        grouped_sections = chunk_by_headings(text)

        # 🛠️ Fallback if no proper headings found
        if not grouped_sections or len(grouped_sections) < 3:
            grouped_sections = fallback_chunk(text)

        texts_to_embed = [section["heading"] + " " + section["content"] for section in grouped_sections]
        embeddings = model.encode(texts_to_embed)

        result = []
        for i, (section, embedding) in enumerate(zip(grouped_sections, embeddings)):
            result.append({
                "chunk_id": f"Collection{collection_num}_{filename}_{i}",
                "collection": collection_num,
                "pdf_name": filename.replace(".txt", ".pdf"),
                "chunk_index": i,
                "heading": section["heading"],
                "text": section["content"],
                "embedding": embedding.tolist()
            })
        return result
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return []

def process_collection_parallel(collection_num):
    start_time = time.time()

    input_dir = f"Collection {collection_num} Extracted/PDFs"
    output_dir = f"Collection {collection_num} Embeddings"
    os.makedirs(output_dir, exist_ok=True)

    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    args_list = [(collection_num, input_dir, filename) for filename in txt_files]

    result = []

    print(f"\n⚙️ Starting multiprocessing for Collection {collection_num} with {len(args_list)} files...")

    with Pool(processes=8, initializer=init_worker) as pool:
        for r in tqdm(pool.imap_unordered(process_file, args_list), total=len(args_list)):
            result.extend(r)

    output_file = os.path.join(output_dir, "chunks.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    elapsed = time.time() - start_time
    print(f"✅ Finished embedding Collection {collection_num} in parallel.")
    print(f"🕒 Time taken for Collection {collection_num}: {elapsed:.2f} seconds ({str(datetime.timedelta(seconds=int(elapsed)))})")

if __name__ == "__main__":
    for collection_number in [1, 2, 3]:
        process_collection_parallel(collection_number)
