import json
import os
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from datetime import datetime
import torch
import time
import re

# ✅ Load local model on CPU
device = torch.device("cpu")
model = SentenceTransformer('./models/minilm', device=device)

# ✅ Utility to determine if a heading is too generic or bad
def is_generic_heading(heading):
    heading = heading.strip().lower()
    unicode_bullets = ["\u2022", "\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005"]

    if any(heading.startswith(bullet) for bullet in unicode_bullets):
        return True
    if heading in ["instructions", "ingredients", "steps", "notes", "procedure", "summary"]:
        return True
    if len(heading.split()) <= 2:
        return True
    return False

# ✅ Remove unicode artifacts like \u2022 from start of strings
def clean_unicode_starts(text):
    # First decode Unicode escape sequences
    try:
        decoded = text.encode().decode('unicode_escape')
    except:
        decoded = text
    
    # Remove various bullet point representations
    cleaned = re.sub(r"\\u00e2\\u0080\\u00a2\s*", "", decoded)
    cleaned = re.sub(r"\u00e2\u0080\u00a2\s*", "", cleaned)
    cleaned = re.sub(r"\\u2022\s*", "", cleaned)
    cleaned = re.sub(r"•\s*", "", cleaned)
    
    # Remove any remaining Unicode patterns
    cleaned = re.sub(r"\\u\w{4}\s*", "", cleaned)
    return cleaned

# ✅ Clean refined text by removing bullet points and formatting
def clean_refined_text(text):
    # First decode any Unicode escape sequences like \u00e9 -> é
    try:
        # Handle Unicode escape sequences
        cleaned = text.encode().decode('unicode_escape')
    except:
        # If decoding fails, use original text
        cleaned = text
    
    # Remove various bullet point representations
    # Remove \u00e2\u0080\u00a2 (UTF-8 encoded bullet)
    cleaned = re.sub(r"\\u00e2\\u0080\\u00a2\s*", "", cleaned)
    # Remove unicode bullet points like \u2022
    cleaned = re.sub(r"\\u2022\s*", "", cleaned)
    # Remove actual bullet characters
    cleaned = re.sub(r"•\s*", "", cleaned)
    # Remove the decoded UTF-8 bullet sequence
    cleaned = re.sub(r"\u00e2\u0080\u00a2\s*", "", cleaned)
    # Remove standalone bullet points at start of lines
    cleaned = re.sub(r"^\s*•\s*", "", cleaned, flags=re.MULTILINE)
    
    # Clean up any remaining Unicode artifacts
    cleaned = re.sub(r"\\u[0-9a-fA-F]{4}", "", cleaned)
    
    # Remove multiple spaces and clean up
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def load_embeddings(collection_num):
    path = f"Collection {collection_num} Embeddings/chunks.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_input(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

def match_document_chunks(document_name, all_chunks):
    return [chunk for chunk in all_chunks if chunk['pdf_name'] == document_name]

def compute_similarity(query, chunks):
    query_emb = model.encode(query, convert_to_tensor=True).to(device)
    scored_chunks = []

    for chunk in chunks:
        chunk_emb = torch.tensor(chunk['embedding'], device=device)
        sim = util.cos_sim(query_emb, chunk_emb)[0][0].item()

        length_penalty = min(len(chunk['text'].split()) / 80, 1.0)
        bullet_penalty = 0.9 if chunk['text'].count("•") > 3 else 1.0
        adjusted_sim = sim * length_penalty * bullet_penalty

        scored_chunks.append((adjusted_sim, chunk))
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return scored_chunks

def find_section_headings(chunks, query):
    """Find chunks that are likely section headings or have good headings"""
    query_emb = model.encode(query, convert_to_tensor=True).to(device)
    heading_chunks = []
    
    for chunk in chunks:
        chunk_emb = torch.tensor(chunk['embedding'], device=device)
        sim = util.cos_sim(query_emb, chunk_emb)[0][0].item()
        
        # Prioritize chunks with good headings
        heading = chunk.get("heading", "").strip()
        has_good_heading = heading and not is_generic_heading(heading) and len(heading.split()) > 2
        
        # Boost score for chunks with good headings
        heading_bonus = 1.3 if has_good_heading else 1.0
        adjusted_sim = sim * heading_bonus
        
        heading_chunks.append((adjusted_sim, chunk))
    
    heading_chunks.sort(reverse=True, key=lambda x: x[0])
    return heading_chunks

def find_content_chunks(chunks, query):
    """Find chunks with substantial content for subsection analysis"""
    query_emb = model.encode(query, convert_to_tensor=True).to(device)
    content_chunks = []
    
    for chunk in chunks:
        chunk_emb = torch.tensor(chunk['embedding'], device=device)
        sim = util.cos_sim(query_emb, chunk_emb)[0][0].item()
        
        # Prioritize chunks with substantial content
        word_count = len(chunk['text'].split())
        content_bonus = 1.2 if word_count > 50 else 1.0
        
        # Bonus for chunks with bullet points (detailed information)
        bullet_bonus = 1.1 if chunk['text'].count("•") > 2 else 1.0
        
        adjusted_sim = sim * content_bonus * bullet_bonus
        content_chunks.append((adjusted_sim, chunk))
    
    content_chunks.sort(reverse=True, key=lambda x: x[0])
    return content_chunks

def extract_output(input_data, chunks_data):
    query = f"{input_data['persona']['role']}: {input_data['job_to_be_done']['task']}"
    metadata = {
        "input_documents": [doc["filename"] for doc in input_data["documents"]],
        "persona": input_data["persona"]["role"],
        "job_to_be_done": input_data["job_to_be_done"]["task"],
        "processing_timestamp": datetime.now().isoformat()
    }

    doc_chunks = defaultdict(list)
    for doc in input_data["documents"]:
        matched = match_document_chunks(doc["filename"], chunks_data)
        if not matched:
            print(f"⚠️ No chunks found for document: {doc['filename']}")
        doc_chunks[doc["filename"]] = matched

    # Separate processing for section headings and content
    all_heading_chunks = []
    all_content_chunks = []
    
    for doc_name, chunks in doc_chunks.items():
        if not chunks:
            continue
            
        # Find best heading chunks per document (top 3)
        heading_scored = find_section_headings(chunks, query)
        all_heading_chunks.extend([(doc_name, s[1], s[0], 'heading') for s in heading_scored[:3]])
        
        # Find best content chunks per document (different chunks, skip first 3)
        content_scored = find_content_chunks(chunks, query)
        # Skip chunks that might overlap with headings by taking different range
        start_idx = 2 if len(content_scored) > 4 else 0
        all_content_chunks.extend([(doc_name, s[1], s[0], 'content') for s in content_scored[start_idx:start_idx+3]])

    # Ensure no overlap between heading and content selections
    used_chunk_ids = set()
    
    # Select top 5 sections for extracted_sections (prioritizing different documents)
    seen_docs_headings = set()
    top_heading_sections = []
    for doc_name, chunk, score, chunk_type in sorted(all_heading_chunks, key=lambda x: x[2], reverse=True):
        chunk_id = f"{doc_name}_{chunk.get('chunk_index', 0)}"
        if (doc_name not in seen_docs_headings and 
            len(top_heading_sections) < 5 and 
            chunk_id not in used_chunk_ids):
            top_heading_sections.append((doc_name, chunk, score))
            seen_docs_headings.add(doc_name)
            used_chunk_ids.add(chunk_id)

    # Fill remaining slots if needed, but avoid already used chunks
    if len(top_heading_sections) < 5:
        for doc_name, chunk, score, chunk_type in sorted(all_heading_chunks, key=lambda x: x[2], reverse=True):
            chunk_id = f"{doc_name}_{chunk.get('chunk_index', 0)}"
            if (len(top_heading_sections) >= 5 or chunk_id in used_chunk_ids):
                continue
            if (doc_name, chunk) not in [(d, c) for d, c, _ in top_heading_sections]:
                top_heading_sections.append((doc_name, chunk, score))
                used_chunk_ids.add(chunk_id)

    # Select top content chunks for subsection_analysis (ensuring no overlap)
    seen_docs_content = set()
    top_content_sections = []
    for doc_name, chunk, score, chunk_type in sorted(all_content_chunks, key=lambda x: x[2], reverse=True):
        chunk_id = f"{doc_name}_{chunk.get('chunk_index', 0)}"
        if (doc_name not in seen_docs_content and 
            len(top_content_sections) < 5 and 
            chunk_id not in used_chunk_ids):
            top_content_sections.append((doc_name, chunk, score))
            seen_docs_content.add(doc_name)

    # Fill remaining slots if needed for content
    if len(top_content_sections) < 5:
        for doc_name, chunk, score, chunk_type in sorted(all_content_chunks, key=lambda x: x[2], reverse=True):
            chunk_id = f"{doc_name}_{chunk.get('chunk_index', 0)}"
            if (len(top_content_sections) >= 5 or chunk_id in used_chunk_ids):
                continue
            if (doc_name, chunk) not in [(d, c) for d, c, _ in top_content_sections]:
                top_content_sections.append((doc_name, chunk, score))

    # Build extracted_sections from heading chunks
    extracted_sections = []
    for i, (doc_name, chunk, _) in enumerate(top_heading_sections):
        heading = clean_unicode_starts(chunk.get("heading", "").strip())

        if not heading or is_generic_heading(heading):
            # Create meaningful preview from content (clean it first)
            clean_text = clean_refined_text(chunk["text"])
            text_words = clean_text.strip().split()
            if text_words:
                # Look for first sentence or meaningful phrase
                first_sentence = clean_text.split('.')[0] if '.' in clean_text else ' '.join(text_words[:15])
                section_title = first_sentence.strip()
                if len(section_title) > 100:
                    section_title = ' '.join(text_words[:15]) + "..."
            else:
                continue
        else:
            section_title = clean_unicode_starts(heading)

        if len(section_title.strip().split()) <= 2:
            continue

        extracted_sections.append({
            "document": doc_name,
            "section_title": section_title,
            "importance_rank": i + 1,
            "page_number": chunk.get("chunk_index", 0) + 1
        })

    # Build subsection_analysis from content chunks (different selection, cleaned text)
    subsection_analysis = []
    for doc_name, chunk, _ in top_content_sections:
        cleaned_text = clean_refined_text(chunk["text"])
        if cleaned_text.strip():  # Only add if there's meaningful content after cleaning
            subsection_analysis.append({
                "document": doc_name,
                "refined_text": cleaned_text,
                "page_number": chunk.get("chunk_index", 0) + 1
            })

    return {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

def generate_output_file(collection_num, input_json_path, output_path):
    start_time = time.time()
    input_data = load_input(input_json_path)
    chunks_data = load_embeddings(collection_num)
    output_json = extract_output(input_data, chunks_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)

    elapsed_time = time.time() - start_time
    print(f"\n🎯 Top Matched Sections:")
    for sec in output_json["extracted_sections"]:
        print(f"- {sec['document']} :: {sec['section_title']}")
    
    print(f"\n📝 Content Analysis Documents:")
    for sub in output_json["subsection_analysis"]:
        print(f"- {sub['document']} :: {len(sub['refined_text'].split())} words")
    
    print(f"\n✅ Output saved to {output_path}")
    print(f"🕒 Time taken: {elapsed_time:.2f} seconds")

# 🧪 Example run
generate_output_file(
    collection_num=2,
    input_json_path="Collection 2/challenge1b_input.json",
    output_path="Collection 2/challenge1b_output.json"
)