import torch
import re
import os
import traceback
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Annotated, List

app = FastAPI(
    title="Insightful Summaries",
    description="Summarization app with cleaning for noisy text.",
    version="2.2.0",
)

templates_dir = "templates"
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
templates = Jinja2Templates(directory=templates_dir)

# Initialize global variables for model and tokenizer
TOKENIZER = None
MODEL = None
DEVICE = None

# IMPORTANT: Pointing to the directory, NOT the .safetensors file
MODEL_PATH = "./my-fine-tuned-bart-summarizer" 

try:
    if not os.path.isdir(MODEL_PATH):
        print(f"--- WARNING --- Model directory not found at: '{MODEL_PATH}'")
    else:
        # Determine execution device
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the tokenizer and model directly
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
        MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
        
        print("✅ Summarization model loaded successfully!")
except Exception as e:
    print(f"❌ CRITICAL: Error loading model: {e}")
    print("Summarization will be disabled.")

# --- Stronger Preprocessing ---
def preprocess_text(text: str) -> str:
    # Remove references [1], (Smith, 2020)
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = re.sub(r"\(\s*[A-Za-z0-9,\s]+(?:19|20)\d{2}\s*\)", "", text)

    # Remove URLs, emails
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove weird non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()

    # Remove leftover symbols
    text = re.sub(r"[^A-Za-z0-9\s\.\,\!\?\;\:]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Postprocessing (clean model output) ---
def clean_summary(summary: str) -> str:
    # Remove garbled characters
    summary = summary.encode("ascii", errors="ignore").decode()

    # Remove repeated phrases like "called opium called opium"
    summary = re.sub(r"\b(\w+\s+\w+)\s+\1\b", r"\1", summary)

    # Collapse multiple spaces
    summary = re.sub(r"\s+", " ", summary).strip()

    return summary

# --- Chunking ---
def chunk_text(text: str, max_words: int = 600) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

@app.get("/", response_class=HTMLResponse)
async def serve_home_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/summarize", response_class=JSONResponse)
async def handle_summarization_request(
    text_to_summarize: Annotated[str, Form()],
    summary_length: Annotated[str, Form()] = "medium"
):
    if not MODEL or not TOKENIZER:
        return JSONResponse(status_code=503, content={"error": "Summarization model not available."})
    if not text_to_summarize.strip():
        return JSONResponse(status_code=400, content={"error": "Input text cannot be empty."})

    # --- Preprocess input ---
    text_to_summarize = preprocess_text(text_to_summarize)
    
    # --- Word Count Validation ---
    word_count = len(text_to_summarize.split())
    if word_count < 20:
        return JSONResponse(
            status_code=400, 
            content={"error": f"Text is too short to summarize ({word_count} words). Please provide at least 20 words."}
        )

    # --- Handle UI strings and set PROPORTIONAL constraints ---
    summary_length = summary_length.lower()

    def get_dynamic_lengths(current_word_count: int, length_type: str):
        if length_type in ["short", "concise"]:
            # Concise: ~15% to 25% of original text
            m_max = min(100, int(current_word_count * 0.25))
            m_min = min(40, max(15, int(current_word_count * 0.15)))
            
        elif length_type == "detailed":
            # Detailed: ~50% to 80% of original text
            m_max = min(500, int(current_word_count * 0.80))
            m_min = min(250, max(60, int(current_word_count * 0.50)))
            
        else: # Medium
            # Medium: ~30% to 45% of original text
            m_max = min(250, int(current_word_count * 0.45))
            m_min = min(120, max(30, int(current_word_count * 0.30)))
            
        # Safety check: ensure max is always strictly greater than min
        return m_min, max(m_min + 5, m_max)

    try:
        # Chunk if too long
        chunks = chunk_text(text_to_summarize, max_words=600)
        chunk_summaries = []
        
        for chunk in chunks:
            c_word_count = len(chunk.split())
            c_min, c_max = get_dynamic_lengths(c_word_count, summary_length)
            
            inputs = TOKENIZER(chunk, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
            
            summary_ids = MODEL.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                min_length=c_min,
                max_length=c_max,
                num_beams=4,
                length_penalty=2.0, 
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            
            summary_text = TOKENIZER.decode(summary_ids[0], skip_special_tokens=True)
            chunk_summaries.append(summary_text)

        # Final pass if multiple chunks were created
        if len(chunk_summaries) > 1:
            combined = " ".join(chunk_summaries)
            comb_word_count = len(combined.split())
            
            f_min, f_max = get_dynamic_lengths(comb_word_count, summary_length)
            
            inputs = TOKENIZER(combined, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
            
            final_summary_ids = MODEL.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                min_length=f_min,
                max_length=f_max,
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            final_raw_summary = TOKENIZER.decode(final_summary_ids[0], skip_special_tokens=True)
        else:
            final_raw_summary = chunk_summaries[0]

        summary = clean_summary(final_raw_summary)
        return JSONResponse(content={"summary": summary})

    except Exception as e:
        traceback.print_exc() 
        print(f"❌ Error during summarization process: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal error during summarization."})