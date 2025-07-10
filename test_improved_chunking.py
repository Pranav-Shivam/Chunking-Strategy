#!/usr/bin/env python3
"""
Test script for improved document chunking system
Compares the improved chunking with the original results
"""

import json
import os
from chunks.chunk_improved import ImprovedDocumentChunkingPipeline

def load_original_results(json_path: str) -> dict:
    """Load original chunking results"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_chunking_quality(chunks: list) -> dict:
    """Analyze the quality of chunking results"""
    if not chunks:
        return {"error": "No chunks found"}
    
    analysis = {
        "total_chunks": len(chunks),
        "token_distribution": [],
        "content_types": {},
        "quality_scores": [],
        "duplicate_chunks": 0,
        "incomplete_chunks": 0,
        "page_coverage": set(),
        "issues": []
    }
    
    chunk_texts = []
    
    for chunk in chunks:
        # Token distribution
        tokens = chunk.get('tokens', 0)
        analysis["token_distribution"].append(tokens)
        
        # Content type analysis
        content_type = chunk.get('content_type', 'unknown')
        analysis["content_types"][content_type] = analysis["content_types"].get(content_type, 0) + 1
        
        # Quality scores
        confidence = chunk.get('confidence_score', 0)
        analysis["quality_scores"].append(confidence)
        
        # Page coverage
        page_start = chunk.get('page_start', 0)
        page_end = chunk.get('page_end', 0)
        analysis["page_coverage"].update(range(page_start, page_end + 1))
        
        # Check for issues
        text = chunk.get('text', '')
        chunk_texts.append(text)
        
        # Check for incomplete chunks
        if tokens < 50:
            analysis["incomplete_chunks"] += 1
            analysis["issues"].append(f"Chunk {chunk.get('chunk_id', 'unknown')} too small: {tokens} tokens")
        
        # Check for low quality
        if confidence < 0.3:
            analysis["issues"].append(f"Chunk {chunk.get('chunk_id', 'unknown')} low quality: {confidence:.2f}")
    
    # Check for duplicates
    unique_texts = set()
    for text in chunk_texts:
        if text in unique_texts:
            analysis["duplicate_chunks"] += 1
        else:
            unique_texts.add(text)
    
    # Calculate statistics
    if analysis["token_distribution"]:
        analysis["avg_tokens"] = sum(analysis["token_distribution"]) / len(analysis["token_distribution"])
        analysis["min_tokens"] = min(analysis["token_distribution"])
        analysis["max_tokens"] = max(analysis["token_distribution"])
    
    if analysis["quality_scores"]:
        analysis["avg_quality"] = sum(analysis["quality_scores"]) / len(analysis["quality_scores"])
        analysis["min_quality"] = min(analysis["quality_scores"])
        analysis["max_quality"] = max(analysis["quality_scores"])
    
    analysis["page_coverage"] = sorted(list(analysis["page_coverage"]))
    
    return analysis

def compare_chunking_results(original_path: str, improved_path: str):
    """Compare original and improved chunking results"""
    print("=" * 80)
    print("CHUNKING QUALITY COMPARISON")
    print("=" * 80)
    
    # Load original results
    try:
        original_data = load_original_results(original_path)
        original_chunks = original_data.get('chunks', [])
        print(f"Original chunks: {len(original_chunks)}")
    except Exception as e:
        print(f"Error loading original results: {e}")
        return
    
    # Load improved results
    try:
        improved_data = load_original_results(improved_path)
        improved_chunks = improved_data.get('chunks', [])
        print(f"Improved chunks: {len(improved_chunks)}")
    except Exception as e:
        print(f"Error loading improved results: {e}")
        return
    
    # Analyze original results
    print("\n" + "=" * 40)
    print("ORIGINAL CHUNKING ANALYSIS")
    print("=" * 40)
    original_analysis = analyze_chunking_quality(original_chunks)
    print_analysis(original_analysis)
    
    # Analyze improved results
    print("\n" + "=" * 40)
    print("IMPROVED CHUNKING ANALYSIS")
    print("=" * 40)
    improved_analysis = analyze_chunking_quality(improved_chunks)
    print_analysis(improved_analysis)
    
    # Comparison summary
    print("\n" + "=" * 40)
    print("IMPROVEMENT SUMMARY")
    print("=" * 40)
    
    if original_analysis.get("total_chunks", 0) > 0 and improved_analysis.get("total_chunks", 0) > 0:
        # Quality improvement
        orig_avg_quality = original_analysis.get("avg_quality", 0)
        impr_avg_quality = improved_analysis.get("avg_quality", 0)
        quality_improvement = ((impr_avg_quality - orig_avg_quality) / orig_avg_quality * 100) if orig_avg_quality > 0 else 0
        
        print(f"Quality Improvement: {quality_improvement:+.1f}%")
        print(f"  Original avg quality: {orig_avg_quality:.3f}")
        print(f"  Improved avg quality: {impr_avg_quality:.3f}")
        
        # Duplicate reduction
        orig_duplicates = original_analysis.get("duplicate_chunks", 0)
        impr_duplicates = improved_analysis.get("duplicate_chunks", 0)
        duplicate_reduction = orig_duplicates - impr_duplicates
        
        print(f"Duplicate Reduction: {duplicate_reduction:+d} chunks")
        print(f"  Original duplicates: {orig_duplicates}")
        print(f"  Improved duplicates: {impr_duplicates}")
        
        # Incomplete chunk reduction
        orig_incomplete = original_analysis.get("incomplete_chunks", 0)
        impr_incomplete = improved_analysis.get("incomplete_chunks", 0)
        incomplete_reduction = orig_incomplete - impr_incomplete
        
        print(f"Incomplete Chunk Reduction: {incomplete_reduction:+d} chunks")
        print(f"  Original incomplete: {orig_incomplete}")
        print(f"  Improved incomplete: {impr_incomplete}")
        
        # Token distribution improvement
        orig_avg_tokens = original_analysis.get("avg_tokens", 0)
        impr_avg_tokens = improved_analysis.get("avg_tokens", 0)
        token_improvement = ((impr_avg_tokens - orig_avg_tokens) / orig_avg_tokens * 100) if orig_avg_tokens > 0 else 0
        
        print(f"Token Distribution Improvement: {token_improvement:+.1f}%")
        print(f"  Original avg tokens: {orig_avg_tokens:.0f}")
        print(f"  Improved avg tokens: {impr_avg_tokens:.0f}")

def print_analysis(analysis: dict):
    """Print analysis results"""
    print(f"Total chunks: {analysis.get('total_chunks', 0)}")
    
    if analysis.get("avg_tokens"):
        print(f"Token distribution:")
        print(f"  Average: {analysis['avg_tokens']:.0f}")
        print(f"  Min: {analysis['min_tokens']}")
        print(f"  Max: {analysis['max_tokens']}")
    
    if analysis.get("avg_quality"):
        print(f"Quality scores:")
        print(f"  Average: {analysis['avg_quality']:.3f}")
        print(f"  Min: {analysis['min_quality']:.3f}")
        print(f"  Max: {analysis['max_quality']:.3f}")
    
    print(f"Content types: {analysis.get('content_types', {})}")
    print(f"Duplicate chunks: {analysis.get('duplicate_chunks', 0)}")
    print(f"Incomplete chunks: {analysis.get('incomplete_chunks', 0)}")
    print(f"Page coverage: {analysis.get('page_coverage', [])}")
    
    if analysis.get("issues"):
        print(f"Issues found: {len(analysis['issues'])}")
        for issue in analysis['issues'][:5]:  # Show first 5 issues
            print(f"  - {issue}")

def test_improved_chunking(pdf_path: str):
    """Test the improved chunking system"""
    print("Testing improved chunking system...")
    
    # Initialize pipeline
    pipeline = ImprovedDocumentChunkingPipeline(
        max_tokens=1000,
        overlap_percentage=0.1
    )
    
    # Process document
    doc_id = "test_doc_001"
    chunks = pipeline.process_document(pdf_path, doc_id)
    
    # Export results
    output_path = "storage/improved_test_results.json"
    pipeline.export_chunks(chunks, output_path)
    
    print(f"Processed {len(chunks)} chunks")
    print(f"Results saved to: {output_path}")
    
    # Analyze results
    analysis = analyze_chunking_quality([chunk.to_dict() for chunk in chunks])
    print_analysis(analysis)
    
    return output_path

if __name__ == "__main__":
    # Test with the BERT PDF
    pdf_path = "BERT_ Conceptual Understanding Explained_.pdf"
    original_json = "BERT_ Conceptual Understanding Explained_.json"
    
    if os.path.exists(pdf_path):
        print("Testing improved chunking system...")
        improved_json = test_improved_chunking(pdf_path)
        
        if os.path.exists(original_json):
            print("\nComparing results...")
            compare_chunking_results(original_json, improved_json)
        else:
            print(f"Original results file not found: {original_json}")
    else:
        print(f"PDF file not found: {pdf_path}")
        print("Please ensure the BERT PDF file is in the current directory.") 