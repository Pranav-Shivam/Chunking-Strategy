<<<<<<< HEAD
# Document Chunking API

A FastAPI service that processes PDF documents and splits them into semantic chunks using advanced NLP techniques.

## Features

- PDF text extraction with multiple fallback methods
- Semantic chunking based on sentence similarity
- Hierarchical document structure preservation
- Token-aware chunking with configurable limits
- Overlap management between chunks
- RESTful API with file upload support

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python run_server.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /upload
Upload a PDF file for chunking.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: PDF file

**Response:**
```json
{
  "message": "File processed successfully",
  "doc_id": "uuid",
  "total_chunks": 10,
  "chunks": [
    {
      "text": "chunk content...",
      "section": "Introduction",
      "chunk_id": "Introduction_1",
      "chunk_index": 0,
      "tokens": 150,
      "page_start": 1,
      "page_end": 1,
      "parent_heading": "Introduction",
      "source_doc_id": "uuid",
      "heading_level": 1
    }
  ]
}
```

### GET /health
Health check endpoint.

## Usage Examples

### Using curl
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

### Using Python requests
```python
import requests

with open("document.pdf", "rb") as f:
    files = {"file": ("document.pdf", f, "application/pdf")}
    response = requests.post("http://localhost:8000/upload", files=files)

if response.status_code == 200:
    result = response.json()
    print(f"Processed {result['total_chunks']} chunks")
```

### Using the test script
```bash
python test_api.py
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Configuration

The chunking pipeline can be configured by modifying the parameters in `main.py`:

- `max_tokens`: Maximum tokens per chunk (default: 1000)
- `overlap_percentage`: Overlap between consecutive chunks (default: 0.15)
- `semantic_model`: Sentence transformer model (default: 'all-MiniLM-L6-v2') 
=======
# Chunking-Strategy
>>>>>>> 68566f7c4a283c145b0e654b9add763e0c5fcc87
