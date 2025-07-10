import re
import json
import nltk
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import pdfplumber
from pdfminer.high_level import extract_text
import pytesseract
from PIL import Image
import io


@dataclass
class ChunkMetadata:
    """Metadata for each chunk"""
    section_name: str
    chunk_id: str
    chunk_index: int
    tokens: int
    page_start: int
    page_end: int
    parent_heading: str
    source_doc_id: str
    heading_level: int = 0


@dataclass
class DocumentChunk:
    """Represents a single document chunk"""
    text: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'section': self.metadata.section_name,
            'chunk_id': self.metadata.chunk_id,
            'chunk_index': self.metadata.chunk_index,
            'tokens': self.metadata.tokens,
            'page_start': self.metadata.page_start,
            'page_end': self.metadata.page_end,
            'parent_heading': self.metadata.parent_heading,
            'source_doc_id': self.metadata.source_doc_id,
            'heading_level': self.metadata.heading_level
        }


class PDFExtractor(ABC):
    """Abstract base class for PDF extraction"""
    
    @abstractmethod
    def extract_text(self, pdf_path: str) -> str:
        pass


class PyMuPDFExtractor(PDFExtractor):
    """PDF extraction using PyMuPDF"""
    
    def extract_text(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text


class PDFMinerExtractor(PDFExtractor):
    """PDF extraction using pdfminer.six"""
    
    def extract_text(self, pdf_path: str) -> str:
        return extract_text(pdf_path)


class PDFPlumberExtractor(PDFExtractor):
    """PDF extraction using pdfplumber"""
    
    def extract_text(self, pdf_path: str) -> str:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text


class OCRExtractor(PDFExtractor):
    """OCR-based PDF extraction using Tesseract"""
    
    def extract_text(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Use OCR to extract text
            page_text = pytesseract.image_to_string(img)
            text += page_text + "\n"
        
        doc.close()
        return text


class TextNormalizer:
    """Handles text normalization and cleaning"""
    
    def __init__(self):
        self.heading_patterns = [
            r'^#{1,6}\s+.+$',  # Markdown headers
            r'^\d+(\.\d+)*\s+[A-Z]',  # Numeric headings
            r'^(Abstract|Introduction|Methodology|Results|Discussion|Conclusion|References|Appendix)\s*$',
            r'^\d+\.\s+[A-Z]'  # Simple numbered sections
        ]
    
    def normalize_text(self, text: str) -> str:
        """Normalize text layout and structure"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Reconstruct paragraphs
        lines = text.split('\n')
        normalized_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                normalized_lines.append('')
            elif self._is_heading(line):
                normalized_lines.append(f"\n{line}\n")
            else:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def _is_heading(self, line: str) -> bool:
        """Check if a line is a heading"""
        for pattern in self.heading_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def convert_to_markdown(self, text: str) -> str:
        """Convert text to markdown format"""
        lines = text.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append('')
                continue
            
            # Convert headings to markdown
            if re.match(r'^\d+\.\s+[A-Z]', line):
                markdown_lines.append(f"# {line}")
            elif re.match(r'^\d+\.\d+\s+[A-Z]', line):
                markdown_lines.append(f"## {line}")
            elif re.match(r'^\d+\.\d+\.\d+\s+[A-Z]', line):
                markdown_lines.append(f"### {line}")
            else:
                markdown_lines.append(line)
        
        return '\n'.join(markdown_lines)


class HierarchicalSplitter:
    """Handles hierarchical document splitting based on sections"""
    
    def __init__(self):
        self.section_patterns = [
            (r'^#{1,6}\s+(.+)$', 1),  # Markdown headers
            (r'^(\d+(\.\d+)*)\s+([A-Z].+)$', 2),  # Numeric headings
            (r'^(Abstract|Introduction|Methodology|Results|Discussion|Conclusion|References|Appendix)\s*$', 1),
            (r'^\d+\.\s+([A-Z].+)$', 1)  # Simple numbered sections
        ]
    
    def split_into_sections(self, text: str) -> List[Dict[str, Any]]:
        """Split text into hierarchical sections"""
        sections = []
        lines = text.split('\n')
        current_section = {'heading': 'Introduction', 'content': [], 'level': 0, 'start_line': 0}
        
        for i, line in enumerate(lines):
            line = line.strip()
            heading_match = self._match_heading(line)
            
            if heading_match:
                # Save previous section
                if current_section['content']:
                    current_section['content'] = '\n'.join(current_section['content'])
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'heading': heading_match['text'],
                    'content': [],
                    'level': heading_match['level'],
                    'start_line': i
                }
            else:
                current_section['content'].append(line)
        
        # Add final section
        if current_section['content']:
            current_section['content'] = '\n'.join(current_section['content'])
            sections.append(current_section)
        
        return sections
    
    def _match_heading(self, line: str) -> Optional[Dict[str, Any]]:
        """Match heading patterns"""
        for pattern, level in self.section_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    return {'text': match.group(2) if match.group(2) else match.group(1), 'level': level}
                else:
                    return {'text': match.group(1), 'level': level}
        return None
    
    def fallback_split(self, text: str) -> List[Dict[str, Any]]:
        """Fallback splitting when no clear headings exist"""
        # Split by multiple newlines
        sections = re.split(r'\n\s*\n\s*\n+', text)
        
        structured_sections = []
        for i, section in enumerate(sections):
            if section.strip():
                structured_sections.append({
                    'heading': f'Section {i+1}',
                    'content': section.strip(),
                    'level': 1,
                    'start_line': i
                })
        
        return structured_sections


class SemanticChunker:
    """Handles semantic chunking within sections"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.75
    
    def chunk_section(self, section_text: str, max_tokens: int = 1000) -> List[str]:
        """Chunk a section based on semantic similarity"""
        sentences = self._split_into_sentences(section_text)
        
        if len(sentences) <= 1:
            return [section_text]
        
        # Generate embeddings
        embeddings = self.model.encode(sentences)
        
        # Find semantic breaks
        chunks = []
        current_chunk = [sentences[0]]
        current_tokens = self._count_tokens(sentences[0])
        
        for i in range(1, len(sentences)):
            similarity = cosine_similarity(
                embeddings[i-1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]
            
            sentence_tokens = self._count_tokens(sentences[i])
            
            # Check if we should start a new chunk
            if (similarity < self.similarity_threshold or 
                current_tokens + sentence_tokens > max_tokens):
                
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentences[i])
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(encoding.encode(text))
        except:
            # Fallback to word count approximation
            return len(text.split()) * 1.3
    
    def fallback_chunk(self, text: str, max_tokens: int = 1000) -> List[str]:
        """Fallback chunking using fixed windows"""
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class TokenController:
    """Handles token control and recursive splitting"""
    
    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.tokenizer = None
    
    def ensure_token_limit(self, text: str) -> List[str]:
        """Ensure text fits within token limits"""
        if self._count_tokens(text) <= self.max_tokens:
            return [text]
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # If single sentence is too long, split it further
            if sentence_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence by clauses
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)
            else:
                if current_tokens + sentence_tokens > self.max_tokens:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split overly long sentences by clauses"""
        # Split by common clause separators
        parts = re.split(r'[;,](?=\s)', sentence)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            part_tokens = self._count_tokens(part)
            
            if current_tokens + part_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [part]
                current_tokens = part_tokens
            else:
                current_chunk.append(part)
                current_tokens += part_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback approximation
            return int(len(text.split()) * 1.3)


class OverlapManager:
    """Handles overlap between chunks"""
    
    def __init__(self, overlap_percentage: float = 0.15):
        self.overlap_percentage = overlap_percentage
    
    def add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk remains unchanged
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i-1]
                overlap_text = self._get_overlap_text(prev_chunk)
                
                # Combine overlap with current chunk
                combined_chunk = f"{overlap_text} {chunk}"
                overlapped_chunks.append(combined_chunk)
        
        return overlapped_chunks
    
    def _get_overlap_text(self, chunk: str) -> str:
        """Get overlap text from end of chunk"""
        sentences = self._split_into_sentences(chunk)
        
        if len(sentences) <= 1:
            return chunk
        
        # Take last 15% of sentences
        overlap_count = max(1, int(len(sentences) * self.overlap_percentage))
        overlap_sentences = sentences[-overlap_count:]
        
        return ' '.join(overlap_sentences)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]


class DocumentChunkingPipeline:
    """Main pipeline orchestrating the entire chunking process"""
    
    def __init__(self, 
                 max_tokens: int = 1000,
                 overlap_percentage: float = 0.15,
                 semantic_model: str = 'all-MiniLM-L6-v2'):
        
        # Initialize components
        self.pdf_extractors = {
            'pymupdf': PyMuPDFExtractor(),
            'pdfminer': PDFMinerExtractor(),
            'pdfplumber': PDFPlumberExtractor(),
            'ocr': OCRExtractor()
        }
        
        self.text_normalizer = TextNormalizer()
        self.hierarchical_splitter = HierarchicalSplitter()
        self.semantic_chunker = SemanticChunker(semantic_model)
        self.token_controller = TokenController(max_tokens)
        self.overlap_manager = OverlapManager(overlap_percentage)
        
        self.max_tokens = max_tokens
    
    def process_document(self, 
                        pdf_path: str, 
                        doc_id: str,
                        extractor_type: str = 'pymupdf') -> List[DocumentChunk]:
        """Process a document through the complete chunking pipeline"""
        
        # Step 1: Extract text from PDF
        raw_text = self._extract_text_with_fallback(pdf_path, extractor_type)
        
        # Step 2: Normalize text
        normalized_text = self.text_normalizer.normalize_text(raw_text)
        markdown_text = self.text_normalizer.convert_to_markdown(normalized_text)
        
        # Step 3: Split into sections
        sections = self._split_into_sections_with_fallback(markdown_text)
        
        # Step 4: Process each section
        all_chunks = []
        
        for section in sections:
            section_chunks = self._process_section(section, doc_id)
            all_chunks.extend(section_chunks)
        
        # Step 5: Add overlap
        chunk_texts = [chunk.text for chunk in all_chunks]
        overlapped_texts = self.overlap_manager.add_overlap(chunk_texts)
        
        # Update chunks with overlapped text
        for i, overlapped_text in enumerate(overlapped_texts):
            if i < len(all_chunks):
                all_chunks[i].text = overlapped_text
                all_chunks[i].metadata.tokens = self.token_controller._count_tokens(overlapped_text)
        
        return all_chunks
    
    def _extract_text_with_fallback(self, pdf_path: str, extractor_type: str) -> str:
        """Extract text with fallback strategies"""
        try:
            return self.pdf_extractors[extractor_type].extract_text(pdf_path)
        except Exception as e:
            # Try fallback extractors
            fallback_order = ['pymupdf', 'pdfplumber', 'pdfminer', 'ocr']
            
            for fallback_type in fallback_order:
                if fallback_type != extractor_type:
                    try:
                        return self.pdf_extractors[fallback_type].extract_text(pdf_path)
                    except Exception:
                        continue
            
            raise Exception(f"All PDF extraction methods failed for {pdf_path}")
    
    def _split_into_sections_with_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Split into sections with fallback"""
        sections = self.hierarchical_splitter.split_into_sections(text)
        
        # If no clear sections found, use fallback
        if len(sections) <= 1:
            sections = self.hierarchical_splitter.fallback_split(text)
        
        return sections
    
    def _process_section(self, section: Dict[str, Any], doc_id: str) -> List[DocumentChunk]:
        """Process a single section into chunks"""
        section_text = section['content']
        
        # Semantic chunking
        try:
            semantic_chunks = self.semantic_chunker.chunk_section(section_text, self.max_tokens)
        except Exception:
            # Fallback to simple chunking
            semantic_chunks = self.semantic_chunker.fallback_chunk(section_text, self.max_tokens)
        
        # Token control
        final_chunks = []
        for chunk_text in semantic_chunks:
            controlled_chunks = self.token_controller.ensure_token_limit(chunk_text)
            final_chunks.extend(controlled_chunks)
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk_text in enumerate(final_chunks):
            metadata = ChunkMetadata(
                section_name=section['heading'],
                chunk_id=f"{section['heading']}_{i+1}",
                chunk_index=i,
                tokens=self.token_controller._count_tokens(chunk_text),
                page_start=1,  # Would need page tracking for accurate values
                page_end=1,
                parent_heading=section['heading'],
                source_doc_id=doc_id,
                heading_level=section['level']
            )
            
            document_chunks.append(DocumentChunk(text=chunk_text, metadata=metadata))
        
        return document_chunks
    
    def export_chunks(self, chunks: List[DocumentChunk], output_path: str) -> None:
        """Export chunks to JSON file"""
        chunk_data = [chunk.to_dict() for chunk in chunks]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)


# Example usage
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = DocumentChunkingPipeline(
        max_tokens=1000,
        overlap_percentage=0.15,
        semantic_model='all-MiniLM-L6-v2'
    )
    
    # Process a document
    try:
        chunks = pipeline.process_document(
            pdf_path="example.pdf",
            doc_id="doc_001",
            extractor_type='pymupdf'
        )
        
        # Export results
        pipeline.export_chunks(chunks, "chunks_output.json")
        
        print(f"Successfully processed document into {len(chunks)} chunks")
        
        # Display first chunk as example
        if chunks:
            print("\nFirst chunk example:")
            print(f"Section: {chunks[0].metadata.section_name}")
            print(f"Tokens: {chunks[0].metadata.tokens}")
            print(f"Text: {chunks[0].text[:200]}...")
            
    except Exception as e:
        print(f"Error processing document: {e}")