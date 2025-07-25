from flask import Flask, request, jsonify
from transformers import BartTokenizer, BartForConditionalGeneration
import logging
import os
import torch
import psutil

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Flask App Init ===
app = Flask(__name__)

# === Check Model Folder ===
logger.info("Files in current directory:")
for item in os.listdir('.'):
    logger.info(f" - {item}")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")
if os.path.exists(MODEL_PATH):
    logger.info(f"Model folder found: {MODEL_PATH}")
    for item in os.listdir(MODEL_PATH):
        logger.info(f" - {item}")
else:
    logger.error(f"Model folder NOT found: {MODEL_PATH}")

# === Load Model and Tokenizer ===
try:
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# === Health Check Endpoint ===
@app.route("/", methods=["GET"])
def home():
    logger.info("HOME API")
    return jsonify({"message": "Summarizer API is running!"})

# === Summarization Endpoint ===
@app.route("/summarize", methods=["POST"])
def summarize():
    logger.info("=== Summarize API called ===")
    try:
        data = request.get_json()
        if "text" not in data or not data["text"].strip():
            logger.error("Missing or empty 'text' in request")
            return jsonify({"error": "Missing or empty 'text' in request"}), 400

        user_text = data["text"].strip()
        logger.info(f"Input text length: {len(user_text)} characters")

        # === Tokenization ===
        inputs = tokenizer(
    user_text,
    return_tensors="pt",
    max_length=256,
    truncation=True,
    padding=True
)

        logger.info(f"Input tensor shape: {inputs.input_ids.shape}")
        
        input_ids = inputs["input_ids"]


        # === Generation ===
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids,
                max_length=100,
                min_length=20,
                num_beams=2,
                early_stopping=True,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

        if not summary or len(summary) < 5:
            summary = "Unable to generate a meaningful summary. The input text may be too short or complex."

        # === Memory Usage ===
        memory_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        logger.info(f"Current program memory usage: {memory_gb:.3f} GB")

        return jsonify({"summary": summary})

    except Exception as e:
        logger.error("=== GENERATION FAILED ===", exc_info=True)
        return jsonify({"error": f"Failed to generate summary: {str(e)}"}), 500

# === Run Server ===
if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(host="0.0.0.0", port=8080)
