from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize Flask app
app = Flask(__name__)

# Load pretrained T5 model and tokenizer once when server starts
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    input_text = "summarize the text i will provide + also return it in documented manner to make it look professional. This is the text input "+ data["text"]

    # Tokenize and generate summary
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=100,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({"summary": summary})

# Run the server
if __name__ == "__main__":
    app.run(debug=True)
