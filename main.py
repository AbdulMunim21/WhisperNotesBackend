from flask import Flask, request, jsonify
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.text_rank import TextRankSummarizer
# from sumy.nlp.stemmers import Stemmer
# from sumy.utils import get_stop_words
import logging
import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("API Key loaded:", os.getenv('OPENAI_API_KEY')[:10] if os.getenv('OPENAI_API_KEY') else "None")
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)

# Initialize summarizer components
# stemmer = Stemmer("english")
# summarizer = TextRankSummarizer(stemmer)
# summarizer.stop_words = get_stop_words("english")

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

        # # Original summarization code (commented out)
        # # Parse and summarize text
        # # parser = PlaintextParser.from_string(user_text, Tokenizer("english"))
        # # 
        # # # Generate summary (default 3 sentences)
        # # # summary_sentences = summarizer(parser.document, 3)
        # # # summary = " ".join([str(sentence) for sentence in summary_sentences])
        # # # summary = summary.strip()
        # # 
        # # logger.info("Summary generated successfully")
        # # logger.info(f"Summary length: {len(summary)} characters")
        # # 
        # # # Validate summary
        # # # if not summary or len(summary) < 5:
        # # #     logger.warning("Generated empty or very short summary")
        # # #     summary = "Unable to generate a meaningful summary. The input text may be too short."

        # GPT-based summarization
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that creates concise, clear summaries of meeting transcripts. Provide summaries in 3-5 bullet points."
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize this meeting transcript:\n\n{user_text}"
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info("Summary generated successfully with GPT")
            logger.info(f"Summary length: {len(summary)} characters")
            
            # Validate summary
            if not summary or len(summary) < 5:
                logger.warning("Generated empty or very short summary")
                summary = "Unable to generate a meaningful summary. The input text may be too short or unclear."
                
        except Exception as gpt_error:
            logger.error(f"GPT API call failed: {str(gpt_error)}")
            summary = "Unable to generate summary due to technical issues. Please try again later."

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