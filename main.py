from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import uuid
from typing import List, Dict, Any
import json
import signal
import sys
import asyncio
import atexit
import weakref

from chunks.chunk_improved import ImprovedDocumentChunkingPipeline

app = FastAPI(title="Document Chunking API", version="1.0.0")

# Initialize the improved chunking pipeline
pipeline = None

def initialize_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = ImprovedDocumentChunkingPipeline(
            max_tokens=1000,
            overlap_percentage=0.1
        )
    return pipeline

# Global cleanup flag
shutting_down = False

# Graceful shutdown handler
def signal_handler(signum, frame):
    global shutting_down
    print("\nShutting down gracefully...")
    shutting_down = True
    
    # Cleanup ML libraries
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    try:
        import numba
        numba.core.dispatcher._dispatcher_cache.clear()
    except:
        pass
    
    # Exit gracefully
    sys.exit(0)

# Cleanup function for atexit
def cleanup_on_exit():
    global shutting_down
    if not shutting_down:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

# Register signal handlers and cleanup
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_on_exit)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Initialize pipeline if not already done
        current_pipeline = initialize_pipeline()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        doc_id = str(uuid.uuid4())
        
        chunks = current_pipeline.process_document(
            pdf_path=temp_file_path,
            doc_id=doc_id
        )
        
        chunk_data = [chunk.to_dict() for chunk in chunks]
        
        # Create storage folder if it doesn't exist
        storage_folder = "storage"
        os.makedirs(storage_folder, exist_ok=True)
        
        # Save JSON file with original filename in storage folder
        base_filename = os.path.splitext(file.filename)[0]
        json_filename = f"{base_filename}.json"
        json_filepath = os.path.join(storage_folder, json_filename)
        
        with open(json_filepath, 'w', encoding='utf-8') as json_file:
            json.dump({
                "filename": file.filename,
                "doc_id": doc_id,
                "total_chunks": len(chunks),
                "chunks": chunk_data
            }, json_file, indent=2, ensure_ascii=False)
        
        os.unlink(temp_file_path)
        
        # json response without chunks
        
        print({
            "message": "File processed successfully",
            "filename": file.filename,
            "doc_id": doc_id,
            "total_chunks": len(chunks),
            "json_saved_as": json_filename
        })
        
        return JSONResponse(content={
            "message": "File processed successfully",
            "filename": file.filename,
            "doc_id": doc_id,
            "total_chunks": len(chunks),
            "json_saved_as": json_filepath,
            "chunks": chunk_data
        })
        
    except Exception as e:
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
