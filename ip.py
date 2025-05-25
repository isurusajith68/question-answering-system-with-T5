import os
import uuid
import torch
import json
from flask import Flask, request, jsonify, Response, stream_with_context, make_response
from flask_cors import CORS, cross_origin
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pdfplumber
from ollama import chat
from prompts import get_mcq_prompt
from groq import Groq
from dotenv import load_dotenv
from pymongo import MongoClient
import random
from bson import ObjectId, json_util
import datetime
import traceback
import logging
from functools import wraps
from typing import Optional, Dict, Any, List
import asyncio
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration class for better organization
class Config:
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://edusmart:13579@edu-smart.s7iq2.mongodb.net")
    GROQ_API_KEY = os.environ.get("Groq_Api_Key")
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'pdf'}
    UPLOAD_FOLDER = './temp_uploads'
    
    # Model configuration
    MAX_CONTEXT_LENGTH = 2048
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0.7

# Initialize Flask app
app = Flask(__name__)

# Configure CORS with more permissive settings for development
CORS(app, 
     origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
     allow_headers=["Content-Type", "Authorization", "Accept"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=True,
     expose_headers=["Content-Disposition"]
)

# Simplified CORS handling - remove the conflicting after_request decorator
@app.after_request
def after_request(response):
    # Allow requests from your frontend origins
    origin = request.headers.get('Origin')
    allowed_origins = ['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:3000', 'http://127.0.0.1:3000']
    
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
    
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,Accept'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response

app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Database connection with error handling
try:
    client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()  # Test connection
    db = client["myDatabase"]
    collection = db["qa_collection"]
    answer_collection = db['answer_collection']
    logger.info("Database connection established successfully")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    raise

# Initialize models with error handling
try:
    model = T5ForConditionalGeneration.from_pretrained("./t5-qna")
    tokenizer = T5Tokenizer.from_pretrained("./t5-qna")
    logger.info("T5 model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load T5 model: {e}")
    model, tokenizer = None, None

# Initialize Groq client
groq_client = None
if Config.GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=Config.GROQ_API_KEY)
        logger.info("Groq client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")

# Utility functions
def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def validate_file_size(file) -> bool:
    """Validate file size."""
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    return size <= Config.MAX_FILE_SIZE

def error_handler(f):
    """Decorator for consistent error handling."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    return decorated_function

def validate_request_data(required_fields: List[str], data: Dict[str, Any]) -> Optional[str]:
    """Validate required fields in request data."""
    missing_fields = [field for field in required_fields if field not in data or not data[field]]
    if missing_fields:
        return f"Missing required fields: {', '.join(missing_fields)}"
    return None

# Improved text processing functions
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF with improved error handling."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
        
        if not text.strip():
            raise ValueError("No readable text found in PDF")
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

def split_text_into_chunks(text: str, max_length: int = Config.MAX_CONTEXT_LENGTH) -> List[str]:
    """Improved text chunking with better boundary detection."""
    if len(text) <= max_length:
        return [text]
    
    # Try splitting by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If single paragraph is too long, split by sentences
        if len(para) > max_length:
            sentences = [s.strip() + '.' for s in para.split('.') if s.strip()]
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 2 <= max_length:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
        else:
            if len(current_chunk) + len(para) + 2 <= max_length:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Fallback to character chunking if needed
    if not chunks:
        chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    
    return chunks

def generate_questions_t5(chunk: str, model, tokenizer) -> str:
    """Generate questions using T5 model with improved prompting."""
    if not model or not tokenizer:
        raise ValueError("T5 model not available")
    
    try:
        input_text = f"Generate a clear, specific question based on this text: {chunk[:1000]}"
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"], 
                max_length=100,
                num_beams=5, 
                early_stopping=True,
                temperature=0.7,
                do_sample=True
            )
        
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Validate generated question
        if not question or len(question.strip()) < 10:
            return "Unable to generate a relevant question from this content."
        
        return question.strip()
    except Exception as e:
        logger.error(f"Error generating question with T5: {e}")
        return "Error generating question."

def predict_answer_t5(context: str, query: str, model, tokenizer) -> str:
    """Generate answers using T5 model with improved prompting."""
    if not model or not tokenizer:
        raise ValueError("T5 model not available")
    
    try:
        input_text = f"Context: {context[:1000]}\nQuestion: {query}\nProvide a concise, accurate answer:"
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"], 
                max_length=150,
                num_beams=5, 
                early_stopping=True,
                temperature=0.3
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if not answer or len(answer.strip()) < 5:
            return "Unable to provide a clear answer."
        
        return answer.strip()
    except Exception as e:
        logger.error(f"Error generating answer with T5: {e}")
        return "Error generating answer."

def generate_mcq_with_groq(chunk: str, question: str) -> Dict[str, Any]:
    """Generate MCQ using Groq API with improved error handling."""
    if not groq_client:
        raise ValueError("Groq client not available")
    
    try:
        messages = get_mcq_prompt(chunk, question)
        
        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=Config.DEFAULT_TEMPERATURE,
            max_tokens=Config.DEFAULT_MAX_TOKENS,
            top_p=0.9,
            stream=False  # Use non-streaming for better error handling
        )
        
        response_content = completion.choices[0].message.content
        
        # Try to parse JSON response
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            # If not JSON, return structured response
            return {
                "question": question,
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": 1,
                "explanation": response_content[:200] + "..." if len(response_content) > 200 else response_content
            }
    
    except Exception as e:
        logger.error(f"Error generating MCQ with Groq: {e}")
        return {
            "question": question,
            "options": ["Unable to generate options"],
            "answer": 1,
            "error": str(e)
        }

def process_pdf_and_generate_qa_stream(pdf_path: str, num_questions: int):
    """Main processing function with streaming response."""
    try:
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text)
        
        # Limit chunks to requested number of questions
        max_chunks = min(len(chunks), num_questions)
        
        for i in range(max_chunks):
            try:
                chunk = chunks[i]
                
                # Generate question using T5
                question = generate_questions_t5(chunk, model, tokenizer)
                
                # Generate MCQ using Groq
                mcq_data = generate_mcq_with_groq(chunk, question)
                
                # Add metadata
                mcq_data.update({
                    "chunk_id": i + 1,
                    "total_chunks": max_chunks,
                    "source_chunk": chunk[:100] + "..." if len(chunk) > 100 else chunk
                })
                
                yield mcq_data
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                yield {
                    "question": f"Error processing question {i+1}",
                    "options": ["Error occurred"],
                    "answer": 1,
                    "error": str(e)
                }
    
    except Exception as e:
        logger.error(f"Error in PDF processing: {e}")
        yield {"error": str(e)}

# Add a simple OPTIONS handler for all routes
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        origin = request.headers.get('Origin')
        allowed_origins = ['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:3000', 'http://127.0.0.1:3000']
        
        if origin in allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
        
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,Accept'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Max-Age'] = '86400'  # Cache preflight for 24 hours
        
        return response

@app.route("/generate-qa", methods=["POST"])
@error_handler
def generate_qa():
    """Generate QA pairs from uploaded PDF."""
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    num_questions = request.form.get("noOfQuestions", 10)
    
    # Validate inputs
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are supported"}), 400
    
    if not validate_file_size(file):
        return jsonify({"error": f"File too large. Maximum size: {Config.MAX_FILE_SIZE // (1024*1024)}MB"}), 400
    
    try:
        num_questions = int(num_questions)
        if num_questions <= 0 or num_questions > 50:
            return jsonify({"error": "Number of questions must be between 1 and 50"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid number of questions"}), 400
    
    # Save file with unique name
    unique_filename = f"{uuid.uuid4().hex}.pdf"
    file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
    
    try:
        file.save(file_path)
        
        def generate():
            try:
                question_count = 0
                for qa_pair in process_pdf_and_generate_qa_stream(file_path, num_questions):
                    question_count += 1
                    logger.info(f"Generated question {question_count}/{num_questions}")
                    yield f"data: {json.dumps(qa_pair)}\n\n"
                
                yield "data: {\"status\": \"complete\"}\n\n"
            
            except Exception as e:
                logger.error(f"Error in generate function: {e}")
                yield f"data: {{\"error\": \"Processing failed: {str(e)}\"}}\n\n"
            
            finally:
                # Cleanup
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up temporary file: {file_path}")
                    except Exception as cleanup_error:
                        logger.error(f"Error during cleanup: {cleanup_error}")
        
        response = Response(
            stream_with_context(generate()), 
            mimetype="text/event-stream"
        )
        
        # Add CORS headers manually to the streaming response
        origin = request.headers.get('Origin')
        allowed_origins = ['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:3000', 'http://127.0.0.1:3000']
        
        if origin in allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
        
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        
        return response
    
    except Exception as e:
        # Cleanup on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

@app.route("/save-qa", methods=["POST"])
@error_handler
def save_qa():
    """Save generated QA data to database."""
    
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Validate required fields
    required_fields = ["name", "time_limit", "questions", "subject", "num_questions"]
    validation_error = validate_request_data(required_fields, data)
    if validation_error:
        return jsonify({"error": validation_error}), 400
    
    # Add metadata
    data.update({
        "complete": False,
        "created_at": datetime.datetime.utcnow(),
        "updated_at": datetime.datetime.utcnow(),
        "version": "1.0"
    })
    
    result = collection.insert_one(data)
    
    return jsonify({
        "message": "QA saved successfully",
        "inserted_id": str(result.inserted_id)
    }), 201

@app.route("/get-qa", methods=["GET"])
@error_handler
def get_qa():
    """Get all incomplete quizzes."""
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    
    skip = (page - 1) * limit
    
    quizzes = list(collection.find(
        {"complete": False},
        {"_id": 1, "name": 1, "time_limit": 1, "questions": 1, "subject": 1, "created_at": 1, "num_questions": 1}
    ).skip(skip).limit(limit))
    
    # Convert ObjectId to string
    for quiz in quizzes:
        quiz["_id"] = str(quiz["_id"])
        quiz["question_count"] = len(quiz.get("questions", []))
    
    total_count = collection.count_documents({"complete": False})
    
    return jsonify({
        "quizzes": quizzes,
        "total": total_count,
        "page": page,
        "limit": limit,
        "total_pages": (total_count + limit - 1) // limit
    }), 200

@app.route("/get-qa/<quiz_id>", methods=["GET"])
@error_handler
def get_single_qa(quiz_id):
    """Get single quiz by ID."""
    if not ObjectId.is_valid(quiz_id):
        return jsonify({"error": "Invalid quiz ID"}), 400
    
    quiz = collection.find_one({"_id": ObjectId(quiz_id)})
    
    if not quiz:
        return jsonify({"error": "Quiz not found"}), 404
    
    quiz["_id"] = str(quiz["_id"])
    quiz["question_length"] = len(quiz.get("questions", []))
    
    return jsonify(quiz), 200

@app.route("/submit-exam", methods=["POST"])
@error_handler
def submit_exam():
    """Submit exam results."""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    required_fields = ["userId", "examId", "score"]
    validation_error = validate_request_data(required_fields, data)
    if validation_error:
        return jsonify({"error": validation_error}), 400
    
    user_id = data["userId"]
    exam_id = data["examId"]
    score = data["score"]
    
    if not ObjectId.is_valid(exam_id):
        return jsonify({"error": "Invalid exam ID"}), 400
    
    # Save exam report
    report_data = {
        "user_id": user_id,
        "exam_id": exam_id,
        "score": score,
        "timestamp": datetime.datetime.utcnow(),
        "answers": data.get("answers", [])  # Store user answers if provided
    }
    
    result = answer_collection.insert_one(report_data)
    
    # Mark exam as complete
    collection.update_one(
        {"_id": ObjectId(exam_id)},
        {"$set": {"complete": True, "updated_at": datetime.datetime.utcnow()}}
    )
    
    return jsonify({
        "message": "Exam submitted successfully",
        "report_id": str(result.inserted_id)
    }), 200

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "services": {
            "database": "connected" if client else "disconnected",
            "groq": "available" if groq_client else "unavailable",
            "t5_model": "loaded" if model and tokenizer else "not_loaded"
        }
    }), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0", 
        port=5005, 
        debug=True,  # Enable debug for development
        threaded=True
    )