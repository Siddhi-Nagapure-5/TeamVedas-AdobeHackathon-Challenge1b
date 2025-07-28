

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Create model folders
os.makedirs("./models/minilm", exist_ok=True)
os.makedirs("./models/distilbart", exist_ok=True)

# Download Sentence Embedding model
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embed_model.save('./models/minilm')

# Download Summarization model
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
summ_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
tokenizer.save_pretrained("./models/distilbart")
summ_model.save_pretrained("./models/distilbart")
