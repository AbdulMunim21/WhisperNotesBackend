from flask import Flask, request, jsonify
from transformers import BartTokenizer, BartForConditionalGeneration
import logging
import os
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

logger.info("Files in current directory:")
for item in os.listdir('.'):
    logger.info(f" - {item}")

# Resolve model path relative to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")

# Debug: Print model contents
logger.info("Checking model folder contents...")
if os.path.exists(MODEL_PATH):
    logger.info(f"Model folder exists: {MODEL_PATH}")
    for item in os.listdir(MODEL_PATH):
        logger.info(f" - {item}")
else:
    logger.error(f"Model folder does NOT exist: {MODEL_PATH}")

# Load model and tokenizer globally (once on server start)
try:
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    logger.info("Model and tokenizer loaded successfully.")
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Optional: Health check route
@app.route("/", methods=["GET"])
def home():
    logger.info("HOME API")
    return jsonify({"message": "Summarizer API is running!"})

@app.route("/summarize", methods=["POST"])
def summarize():
    logger.info("=== Summarize API called ===")

    try:
        data = request.get_json()
        if "text" not in data:
            logger.error("Missing 'text' in request")
            return jsonify({"error": "Missing 'text' in request"}), 400

        user_text = data["text"].strip()
        if len(user_text) == 0:
            logger.error("Input text is empty")
            return jsonify({"error": "Input text is empty"}), 400

        logger.info(f"Input text length: {len(user_text)} characters")

        # ✅ T5 requires task prefix
        prompt = "summarize: " + user_text
        logger.info(f"Prompt: {prompt[:100]}...")  # Log first 100 chars

        # Tokenize with conservative limits for tiny model
        inputs = tokenizer(
            user_text, 
            return_tensors="pt", 
            max_length=256,         # Optimal for this model size
            truncation=True,
            padding=True
        )
        logger.info(f"Input tensor shape: {inputs.input_ids.shape}")

        # Generate with lightweight settings
        logger.info("Starting model generation...")
        with torch.no_grad():
            summary_ids = model.generate(
                inputs.input_ids,
                max_length=100,         # Reasonable output length
                min_length=20,          # Ensure meaningful summary
                num_beams=2,            # Light beam search
                early_stopping=True,
                length_penalty=2.0,     # Encourage good length
                no_repeat_ngram_size=3, # Prevent repetition
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary = summary.strip()
        
        logger.info("Summary generated successfully")
        logger.info(f"Summary length: {len(summary)} characters")

        # Validate summary
        if not summary or len(summary) < 5:
            logger.warning("Generated empty or very short summary")
            summary = "Unable to generate a meaningful summary. The input text may be too short or complex."

        return jsonify({"summary": summary})

    except Exception as e:
        logger.error(f"=== GENERATION FAILED ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return jsonify({"error": f"Failed to generate summary: {str(e)}"}), 500

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=8080)