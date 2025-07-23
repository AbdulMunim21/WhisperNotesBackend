from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
import os

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
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    logger.info("Model and tokenizer loaded successfully.")
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
    logger.info("Summarize API")

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    user_text = data["text"].strip()

    if len(user_text) == 0:
        return jsonify({"error": "Input text is empty"}), 400

    prompt = (
        "Just include the summary and nothing else. summarize the following text in a professional and well-documented format: "
        + user_text
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return jsonify({"summary": summary})

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=8080)