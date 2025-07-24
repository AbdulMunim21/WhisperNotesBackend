from flask import Flask, request, jsonify
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize summarizer components
stemmer = Stemmer("english")
summarizer = TextRankSummarizer(stemmer)
summarizer.stop_words = get_stop_words("english")

logger.info("Sumy summarizer initialized successfully.")

# Health check route
@app.route("/", methods=["GET"])
def home():
    logger.info("HOME API")
    return jsonify({"message": "Sumy Summarizer API is running!"})

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

        # Parse and summarize text
        parser = PlaintextParser.from_string(user_text, Tokenizer("english"))
        
        # Generate summary (default 3 sentences)
        summary_sentences = summarizer(parser.document, 3)
        summary = " ".join([str(sentence) for sentence in summary_sentences])
        summary = summary.strip()
        
        logger.info("Summary generated successfully")
        logger.info(f"Summary length: {len(summary)} characters")

        # Validate summary
        if not summary or len(summary) < 5:
            logger.warning("Generated empty or very short summary")
            summary = "Unable to generate a meaningful summary. The input text may be too short."

        return jsonify({"summary": summary})

    except Exception as e:
        logger.error(f"=== GENERATION FAILED ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return jsonify({"error": f"Failed to generate summary: {str(e)}"}), 500

if __name__ == "__main__":
    print("Starting Flask server with Sumy...")
    app.run(host="0.0.0.0", port=8080)