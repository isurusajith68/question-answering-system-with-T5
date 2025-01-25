import os
import uuid
import torch
import json
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pdfplumber 
from ollama import chat
from prompts import get_mcq_prompt 

app = Flask(__name__)

CORS(app)

model = T5ForConditionalGeneration.from_pretrained("./t5-qna")
tokenizer = T5Tokenizer.from_pretrained("./t5-qna")


def predict_using_ollama(chunk, question):

    try:
        messages =get_mcq_prompt(chunk, question)

        stream = chat(model='llama2', messages=messages, stream=True)
        response = ""
        for chunk in stream:
            content = chunk.get('message', {}).get('content', '')
            if content:
                response += content

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response "}

    except Exception as e:
        return {"error": f"Error generating : {str(e)}"}


def extract_text_from_pdf(pdf_path):

    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        raise


def generate_questions(chunk, model, tokenizer):

    try:
        input_text = f"Given the following text, generate a concise and relevant question: {chunk}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)

        question = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not question or question.lower() in chunk.lower():
            question = "Could not generate a relevant question."

        return question
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        raise


def predict(context, query, model, tokenizer):

    try:
        input_text = f"Given the context: {context}, and the question: {query}, generate a precise and relevant answer."
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=8, early_stopping=True)

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if answer.lower() == query.lower() or not answer:
            answer = "The answer is unclear, please try again."

        return answer
    except Exception as e:
        print(f"Error predicting answer: {str(e)}")
        raise


def process_pdf_and_generate_questions_with_context_stream(pdf_path, model, tokenizer, max_context_length=1024):
 
    try:
        text = extract_text_from_pdf(pdf_path)

        if not text:
            raise ValueError("No text extracted from the PDF. The file may be empty or unsupported.")

        chunks = [text[i:i + max_context_length] for i in range(0, len(text), max_context_length)]

        for i, chunk in enumerate(chunks[:5]): 
            question = generate_questions(chunk, model, tokenizer)

            answer = predict(chunk, question, model, tokenizer)

            refined_answer = predict_using_ollama(chunk, question)

            yield refined_answer

    except Exception as e:
        print(f"Error processing PDF and generating QA: {e}")
        yield {"error": str(e)}


@app.route("/generate-qa", methods=["POST"])
def generate_qa():
    file_path = None 

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    unique_filename = f"{uuid.uuid4().hex}.pdf"
    file_path = os.path.join("./", unique_filename)

    try:
        file.save(file_path)

        if not os.path.exists(file_path):
            return jsonify({"error": "Failed to save file"}), 500

        def generate():
            try:
                for qa_pair in process_pdf_and_generate_questions_with_context_stream(file_path, model, tokenizer):
                    yield f"data: {json.dumps(qa_pair)}\n\n"

                yield "data: {\"status\": \"complete\"}\n\n"
            except Exception as e:
                yield f"data: {{\"error\": \"Failed to process PDF and generate QA\"}}\n\n"
            finally:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as cleanup_error:
                        print(f"Error during cleanup: {cleanup_error}")

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except Exception as e:
        print(f"Error in /generate-qa route: {str(e)}")
        return jsonify({"error": "An error occurred while processing the file."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)