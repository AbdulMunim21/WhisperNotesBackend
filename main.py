from flask import Flask, request, jsonify
import logging
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
import psutil

load_dotenv()

# === Configuration ===
api_key = os.getenv('OPENAI_API_KEY')

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("API Key loaded:", api_key[:10] if api_key else "None")

if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable!")

# === Flask App Setup ===
app = Flask(__name__)

# Simple in-memory cache (for single instance only)
_summary_cache = {}

def get_cache(key):
    """Simple in-memory cache getter"""
    if key in _summary_cache:
        item, timestamp = _summary_cache[key]
        if time.time() - timestamp < 300:  # 5 minutes cache
            return item
        else:
            del _summary_cache[key]
    return None

def set_cache(key, value):
    """Simple in-memory cache setter"""
    _summary_cache[key] = (value, time.time())

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# === Health check route ===
@app.route("/", methods=["GET"])
def home():
    logger.info("HOME API")
    return jsonify({"message": "Summarizer API is running!", "status": "healthy"})

# === Health check endpoint for Railway ===
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

# === Rate limiting helper ===
_rate_limit_store = {}

def is_rate_limited(ip_address, limit=10, window=60):
    """Simple rate limiting implementation"""
    cache_key = f"rate_limit:{ip_address}"
    current_time = time.time()
    
    if cache_key not in _rate_limit_store:
        _rate_limit_store[cache_key] = {"count": 1, "timestamp": current_time}
        return False
    
    item = _rate_limit_store[cache_key]
    if current_time - item["timestamp"] > window:
        # Reset window
        _rate_limit_store[cache_key] = {"count": 1, "timestamp": current_time}
        return False
    elif item["count"] < limit:
        item["count"] += 1
        return False
    else:
        return True

# === Summarization with caching ===
@app.route("/summarize", methods=["POST"])
def summarize():
    start_time = time.time()
    logger.info("=== Summarize API called ===")
    
    try:
        # Rate limiting
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        if is_rate_limited(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
        
        # Validate input
        data = request.get_json()
        if not data or "text" not in data or not data["text"].strip():
            logger.error("Missing or empty 'text' in request")
            return jsonify({"error": "Missing or empty 'text' in request"}), 400

        user_text = data["text"].strip()
        text_length = len(user_text)
        logger.info(f"Input text length: {text_length} characters")
        
        # Check if text is too long
        if text_length > 50000:  # ~10k words
            logger.warning("Text too long for processing")
            return jsonify({"error": "Text is too long. Please provide text under 50,000 characters."}), 400
        
        # Create cache key
        cache_key = f"summary:{hash(user_text[:1000])}"  # Hash first 1000 chars
        cached_summary = get_cache(cache_key)
        
        if cached_summary:
            logger.info("Returning cached summary")
            response_time = time.time() - start_time
            return jsonify({
                "summary": cached_summary,
                "cached": True,
                "response_time": round(response_time, 3)
            })
        
        # Generate summary with OpenAI
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
                temperature=0.3,
                timeout=30  # Add timeout
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info("Summary generated successfully with GPT")
            logger.info(f"Summary length: {len(summary)} characters")
            
            # Validate summary
            if not summary or len(summary) < 5:
                logger.warning("Generated empty or very short summary")
                summary = "Unable to generate a meaningful summary. The input text may be too short or unclear."
            
            # Cache the result for 5 minutes
            set_cache(cache_key, summary)
                
        except Exception as gpt_error:
            logger.error(f"GPT API call failed: {str(gpt_error)}")
            return jsonify({"error": "Unable to generate summary due to technical issues. Please try again later."}), 503

        # === Performance Metrics ===
        response_time = time.time() - start_time
        memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        
        logger.info(f"Response time: {response_time:.3f}s")
        logger.info(f"Memory usage: {memory_mb:.2f} MB")

        return jsonify({
            "summary": summary,
            "cached": False,
            "response_time": round(response_time, 3),
            "memory_usage_mb": round(memory_mb, 2)
        })

    except Exception as e:
        logger.error("=== GENERATION FAILED ===", exc_info=True)
        return jsonify({"error": f"Failed to generate summary: {str(e)}"}), 500

# === Error handlers ===
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error("Internal server error", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("Starting Flask server...")
    # For development only - Railway will use gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 8080)))