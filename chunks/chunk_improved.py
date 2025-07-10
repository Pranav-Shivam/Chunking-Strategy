import re
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import torch
import warnings
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import pdfplumber
from PIL import Image
import io
import tiktoken
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import hashlib
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from collections import defaultdict
import networkx as nx

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt')
    except:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords')
    except:
        pass

# Suppress warnings
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="numba")
warnings.filterwarnings("ignore", message=".*Numba.*")

@dataclass
class ImprovedChunkMetadata:
    section_name: str
    chunk_id: str
    chunk_index: int
    tokens: int
    page_start: int
    page_end: int
    parent_heading: str
    source_doc_id: str
    heading_level: int = 0
    content_type: str = "text"
    confidence_score: float = 0.0
    semantic_topic: str = ""
    word_count: int = 0
    sentence_count: int = 0
    is_heading: bool = False
    is_conclusion: bool = False
    is_introduction: bool = False
    is_reference: bool = False
    is_table: bool = False
    is_figure: bool = False
    is_formula: bool = False
    coherence_score: float = 0.0
    completeness_score: float = 0.0
    table_data: Optional[Dict] = None
    image_refs: List[str] = None
    formula_refs: List[str] = None

@dataclass
class ImprovedDocumentChunk:
    text: str
    metadata: ImprovedChunkMetadata
    embeddings: Optional[np.ndarray] = None
    
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
            'heading_level': self.metadata.heading_level,
            'content_type': self.metadata.content_type,
            'confidence_score': self.metadata.confidence_score,
            'semantic_topic': self.metadata.semantic_topic,
            'word_count': self.metadata.word_count,
            'sentence_count': self.metadata.sentence_count,
            'is_heading': self.metadata.is_heading,
            'is_conclusion': self.metadata.is_conclusion,
            'is_introduction': self.metadata.is_introduction,
            'is_reference': self.metadata.is_reference,
            'is_table': self.metadata.is_table,
            'is_figure': self.metadata.is_figure,
            'is_formula': self.metadata.is_formula,
            'coherence_score': self.metadata.coherence_score,
            'completeness_score': self.metadata.completeness_score,
            'table_data': self.metadata.table_data,
            'image_refs': self.metadata.image_refs or [],
            'formula_refs': self.metadata.formula_refs or []
        }

class TextProcessor:
    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # Try to load stopwords with fallback
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # Fallback: basic English stopwords
            self.stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
            }
        
        # Try to load spaCy model for better text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK with fallback"""
        try:
            return sent_tokenize(text)
        except LookupError:
            # Fallback: simple sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove page numbers and headers
        text = re.sub(r'\b\d+\s*$', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        return text
    
    def detect_heading(self, text: str) -> bool:
        """Detect if text is a heading"""
        if not text.strip():
            return False
        
        # Check for heading patterns
        heading_patterns = [
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^\d+\.\s+[A-Z]',   # Numbered headings
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title Case
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*$'  # Title Case with spaces
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text.strip()):
                return True
        
        # Check if text is short and contains key terms
        words = text.split()
        if len(words) <= 8 and any(word in text.lower() for word in ['introduction', 'conclusion', 'chapter', 'section', 'part']):
            return True
        
        return False
    
    def detect_conclusion(self, text: str) -> bool:
        """Detect if text is a conclusion"""
        conclusion_keywords = ['conclusion', 'summary', 'in conclusion', 'finally', 'thus', 'therefore']
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in conclusion_keywords)
    
    def detect_introduction(self, text: str) -> bool:
        """Detect if text is an introduction"""
        intro_keywords = ['introduction', 'overview', 'background', 'context']
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in intro_keywords)
    
    def detect_reference(self, text: str) -> bool:
        """Detect if text contains references"""
        ref_patterns = [
            r'https?://',
            r'www\.',
            r'\.com',
            r'\.org',
            r'\.edu',
            r'accessed',
            r'reference',
            r'bibliography'
        ]
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in ref_patterns)
    
    def calculate_coherence(self, sentences: List[str]) -> float:
        """Calculate semantic coherence of sentences"""
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence based on word overlap
        total_overlap = 0
        comparisons = 0
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                try:
                    words1 = set(word_tokenize(sentences[i].lower()))
                    words2 = set(word_tokenize(sentences[j].lower()))
                except LookupError:
                    # Fallback: simple word splitting
                    words1 = set(sentences[i].lower().split())
                    words2 = set(sentences[j].lower().split())
                
                if words1 and words2:
                    overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                    total_overlap += overlap
                    comparisons += 1
        
        return total_overlap / comparisons if comparisons > 0 else 1.0
    
    def calculate_completeness(self, text: str) -> float:
        """Calculate completeness score of text"""
        sentences = self.split_into_sentences(text)
        if not sentences:
            return 0.0
        
        # Check for complete sentences
        complete_sentences = 0
        for sentence in sentences:
            if len(sentence.strip()) > 10 and sentence.strip().endswith(('.', '!', '?')):
                complete_sentences += 1
        
        return complete_sentences / len(sentences) if sentences else 0.0

class DocumentStructureAnalyzer:
    def __init__(self):
        self.layout_processor = None
        self.layout_model = None
        self.fallback_mode = False
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize layout analysis models with fallback"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
                self.layout_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
        except Exception:
            self.layout_processor = None
            self.layout_model = None
            self.fallback_mode = True
    
    def analyze_document_structure(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze document structure with proper page tracking"""
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return self._fallback_structure_analysis(pdf_path)
        
        structure = {
            'pages': [],
            'sections': [],
            'tables': [],
            'figures': [],
            'formulas': []
        }
        
        current_section = None
        section_content = []
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                page_dict = page.get_text("dict")
                
                page_structure = self._analyze_page_structure(page, page_dict, page_num)
                structure['pages'].append(page_structure)
                
                # Extract text blocks with proper page tracking
                text_blocks = self._extract_text_blocks(page_dict, page_num)
                
                for block in text_blocks:
                    # Detect section boundaries
                    if self._is_section_heading(block['text']):
                        # Save previous section
                        if current_section and section_content:
                            structure['sections'].append({
                                'title': current_section,
                                'content': section_content,
                                'page_start': section_content[0]['page'],
                                'page_end': section_content[-1]['page']
                            })
                        
                        # Start new section
                        current_section = block['text']
                        section_content = [block]
                    else:
                        if current_section:
                            section_content.append(block)
                        else:
                            # Content before first section
                            section_content.append(block)
                
            except Exception:
                continue
        
        # Add final section
        if current_section and section_content:
            structure['sections'].append({
                'title': current_section,
                'content': section_content,
                'page_start': section_content[0]['page'],
                'page_end': section_content[-1]['page']
            })
        
        doc.close()
        return structure
    
    def _analyze_page_structure(self, page, page_dict, page_num):
        """Analyze individual page structure"""
        blocks = page_dict.get('blocks', [])
        text_blocks = []
        
        for block in blocks:
            if 'lines' in block:
                for line in block['lines']:
                    for span in line['spans']:
                        text_blocks.append({
                            'text': span['text'],
                            'bbox': span['bbox'],
                            'font': span['font'],
                            'size': span['size'],
                            'flags': span['flags'],
                            'page': page_num
                        })
        
        return {
            'page_num': page_num,
            'text_blocks': text_blocks,
            'layout_analysis': self._classify_layout_elements(text_blocks)
        }
    
    def _extract_text_blocks(self, page_dict, page_num):
        """Extract text blocks with proper page tracking"""
        blocks = page_dict.get('blocks', [])
        text_blocks = []
        
        for block in blocks:
            if 'lines' in block:
                block_text = []
                for line in block['lines']:
                    for span in line['spans']:
                        block_text.append(span['text'])
                
                if block_text:
                    text_blocks.append({
                        'text': ' '.join(block_text),
                        'bbox': block['bbox'],
                        'page': page_num,
                        'font_size': max(span['size'] for line in block['lines'] for span in line['spans']) if block['lines'] else 12
                    })
        
        return text_blocks
    
    def _is_section_heading(self, text: str) -> bool:
        """Detect if text is a section heading"""
        if not text.strip():
            return False
        
        # Check for heading patterns
        heading_patterns = [
            r'^\d+\.\s+[A-Z]',  # Numbered sections
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS headings
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title Case
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text.strip()):
                return True
        
        # Check for specific heading keywords
        heading_keywords = ['introduction', 'conclusion', 'chapter', 'section', 'part', 'overview', 'background']
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in heading_keywords)
    
    def _classify_layout_elements(self, text_blocks):
        """Classify layout elements"""
        headings = []
        paragraphs = []
        
        for block in text_blocks:
            if block['flags'] & 2**4:  # Bold
                headings.append(block)
            else:
                paragraphs.append(block)
        
        return {'headings': headings, 'paragraphs': paragraphs}
    
    def _fallback_structure_analysis(self, pdf_path: str) -> Dict[str, Any]:
        """Fallback structure analysis"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                structure = {
                    'pages': [],
                    'sections': [],
                    'tables': [],
                    'figures': [],
                    'formulas': []
                }
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        structure['pages'].append({
                            'page_num': page_num,
                            'text_blocks': [{'text': text, 'bbox': [0, 0, 100, 100], 'page': page_num}],
                            'layout_analysis': {'headings': [], 'paragraphs': []}
                        })
                
                return structure
        except Exception:
            return {
                'pages': [],
                'sections': [],
                'tables': [],
                'figures': [],
                'formulas': []
            }

class SemanticChunker:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.embedding_model = None
        self.text_processor = TextProcessor()
        self.fallback_mode = False
        
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize embedding model with fallback"""
        embedding_models = [
            'all-MiniLM-L6-v2',
            'all-mpnet-base-v2',
            'paraphrase-MiniLM-L6-v2'
        ]
        
        for model_name in embedding_models:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.embedding_model = SentenceTransformer(model_name, device=self.device)
                break
            except Exception:
                continue
        
        if self.embedding_model is None:
            self.fallback_mode = True
    
    def chunk_by_semantic_boundaries(self, text: str, max_tokens: int = 1000) -> List[str]:
        """Chunk text by semantic boundaries"""
        sentences = self.text_processor.split_into_sentences(text)
        if not sentences:
            return []
        
        # Clean sentences
        sentences = [self.text_processor.clean_text(s) for s in sentences if self.text_processor.clean_text(s)]
        
        if len(sentences) == 1:
            return [sentences[0]]
        
        # Get embeddings for semantic similarity
        if not self.fallback_mode:
            try:
                embeddings = self.embedding_model.encode(sentences)
                boundaries = self._find_semantic_boundaries(embeddings, max_tokens)
            except Exception:
                boundaries = self._fallback_boundaries(sentences, max_tokens)
        else:
            boundaries = self._fallback_boundaries(sentences, max_tokens)
        
        # Create chunks based on boundaries
        chunks = []
        start_idx = 0
        
        for boundary in boundaries:
            chunk_sentences = sentences[start_idx:boundary]
            chunk_text = ' '.join(chunk_sentences)
            
            if len(chunk_text.strip()) > 50:  # Minimum viable chunk
                chunks.append(chunk_text)
            
            start_idx = boundary
        
        # Add remaining sentences
        if start_idx < len(sentences):
            remaining_text = ' '.join(sentences[start_idx:])
            if len(remaining_text.strip()) > 50:
                chunks.append(remaining_text)
        
        return chunks
    
    def _find_semantic_boundaries(self, embeddings: np.ndarray, max_tokens: int) -> List[int]:
        """Find semantic boundaries using embeddings"""
        boundaries = []
        current_tokens = 0
        
        for i in range(1, len(embeddings)):
            # Calculate similarity between consecutive sentences
            similarity = cosine_similarity(
                embeddings[i-1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]
            
            # Estimate tokens for current sentence
            sentence_tokens = len(self.text_processor.tokenizer.encode(embeddings[i]))
            current_tokens += sentence_tokens
            
            # Break if similarity is low or token limit exceeded
            if similarity < 0.7 or current_tokens > max_tokens:
                boundaries.append(i)
                current_tokens = sentence_tokens
        
        return boundaries
    
    def _fallback_boundaries(self, sentences: List[str], max_tokens: int) -> List[int]:
        """Fallback boundary detection"""
        boundaries = []
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(self.text_processor.tokenizer.encode(sentence))
            current_tokens += sentence_tokens
            
            if current_tokens > max_tokens:
                boundaries.append(i)
                current_tokens = sentence_tokens
        
        return boundaries

class QualityOptimizer:
    def __init__(self):
        self.text_processor = TextProcessor()
    
    def optimize_chunks(self, chunks: List[str]) -> List[str]:
        """Optimize chunks for quality and coherence"""
        optimized_chunks = []
        
        for chunk in chunks:
            # Clean the chunk
            cleaned_chunk = self.text_processor.clean_text(chunk)
            
            if self._is_viable_chunk(cleaned_chunk):
                optimized_chunks.append(cleaned_chunk)
        
        # Remove duplicates
        optimized_chunks = self._remove_duplicates(optimized_chunks)
        
        return optimized_chunks
    
    def _is_viable_chunk(self, chunk: str) -> bool:
        """Check if chunk is viable"""
        if not chunk.strip():
            return False
        
        # Check minimum size
        tokens = len(self.text_processor.tokenizer.encode(chunk))
        if tokens < 50:
            return False
        
        # Check for complete sentences
        sentences = self.text_processor.split_into_sentences(chunk)
        if len(sentences) < 1:
            return False
        
        return True
    
    def _remove_duplicates(self, chunks: List[str]) -> List[str]:
        """Remove duplicate or highly similar chunks"""
        unique_chunks = []
        
        for chunk in chunks:
            is_duplicate = False
            
            for existing_chunk in unique_chunks:
                # Calculate similarity
                similarity = self._calculate_text_similarity(chunk, existing_chunk)
                if similarity > 0.8:  # High similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        words1 = set(word_tokenize(text1.lower()))
        words2 = set(word_tokenize(text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_chunk_quality(self, chunk: str) -> float:
        """Calculate overall chunk quality"""
        sentences = self.text_processor.split_into_sentences(chunk)
        
        # Coherence score
        coherence = self.text_processor.calculate_coherence(sentences)
        
        # Completeness score
        completeness = self.text_processor.calculate_completeness(chunk)
        
        # Length score (prefer chunks around 1000 tokens)
        tokens = len(self.text_processor.tokenizer.encode(chunk))
        length_score = 1.0 - abs(tokens - 1000) / 1000  # Normalize around 1000 tokens
        
        # Combine scores
        quality_score = (coherence * 0.4 + completeness * 0.4 + length_score * 0.2)
        
        return max(0.0, min(1.0, quality_score))

class ImprovedDocumentChunkingPipeline:
    def __init__(self, 
                 max_tokens: int = 1000,
                 overlap_percentage: float = 0.1,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.max_tokens = max_tokens
        self.overlap_percentage = overlap_percentage
        self.device = device
        
        # Initialize components
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.semantic_chunker = SemanticChunker(device)
        self.quality_optimizer = QualityOptimizer()
        self.text_processor = TextProcessor()
    
    def process_document(self, pdf_path: str, doc_id: str) -> List[ImprovedDocumentChunk]:
        """Process document with improved chunking"""
        # Analyze document structure
        structure = self.structure_analyzer.analyze_document_structure(pdf_path)
        
        all_chunks = []
        chunk_index = 0
        
        # Process each section
        for section in structure['sections']:
            section_chunks = self._process_section(section, doc_id, chunk_index)
            all_chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        # If no sections found, process as single document
        if not structure['sections']:
            all_chunks = self._process_entire_document(pdf_path, doc_id)
        
        # Optimize chunks
        all_chunks = self._optimize_chunks(all_chunks)
        
        return all_chunks
    
    def _process_section(self, section: Dict, doc_id: str, start_index: int) -> List[ImprovedDocumentChunk]:
        """Process individual section"""
        section_title = section['title']
        section_content = section['content']
        
        # Combine section content
        full_text = ' '.join([block['text'] for block in section_content])
        
        # Chunk by semantic boundaries
        text_chunks = self.semantic_chunker.chunk_by_semantic_boundaries(full_text, self.max_tokens)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Analyze chunk properties
            sentences = self.text_processor.split_into_sentences(chunk_text)
            word_count = len(chunk_text.split())
            sentence_count = len(sentences)
            tokens = len(self.text_processor.tokenizer.encode(chunk_text))
            
            # Detect chunk type
            is_heading = self.text_processor.detect_heading(chunk_text)
            is_conclusion = self.text_processor.detect_conclusion(chunk_text)
            is_introduction = self.text_processor.detect_introduction(chunk_text)
            is_reference = self.text_processor.detect_reference(chunk_text)
            
            # Calculate quality scores
            coherence_score = self.text_processor.calculate_coherence(sentences)
            completeness_score = self.text_processor.calculate_completeness(chunk_text)
            quality_score = self.quality_optimizer.calculate_chunk_quality(chunk_text)
            
            # Determine content type
            content_type = self._determine_content_type(chunk_text, is_heading, is_reference)
            
            # Get page range
            page_start = section['page_start']
            page_end = section['page_end']
            
            # Create metadata
            metadata = ImprovedChunkMetadata(
                section_name=section_title,
                chunk_id=f"{section_title}_{start_index + i}",
                chunk_index=start_index + i,
                tokens=tokens,
                page_start=page_start,
                page_end=page_end,
                parent_heading=section_title,
                source_doc_id=doc_id,
                heading_level=1 if is_heading else 0,
                content_type=content_type,
                confidence_score=quality_score,
                semantic_topic=f"Topic_{start_index + i}",
                word_count=word_count,
                sentence_count=sentence_count,
                is_heading=is_heading,
                is_conclusion=is_conclusion,
                is_introduction=is_introduction,
                is_reference=is_reference,
                is_table=False,
                is_figure=False,
                is_formula=False,
                coherence_score=coherence_score,
                completeness_score=completeness_score
            )
            
            # Create document chunk
            document_chunk = ImprovedDocumentChunk(
                text=chunk_text,
                metadata=metadata
            )
            
            chunks.append(document_chunk)
        
        return chunks
    
    def _process_entire_document(self, pdf_path: str, doc_id: str) -> List[ImprovedDocumentChunk]:
        """Process entire document when no sections are detected"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                page_ranges = []
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        full_text += text + "\n"
                        page_ranges.append(page_num)
                
                if not full_text.strip():
                    return []
                
                # Chunk the entire text
                text_chunks = self.semantic_chunker.chunk_by_semantic_boundaries(full_text, self.max_tokens)
                
                chunks = []
                for i, chunk_text in enumerate(text_chunks):
                    # Analyze chunk properties
                    sentences = self.text_processor.split_into_sentences(chunk_text)
                    word_count = len(chunk_text.split())
                    sentence_count = len(sentences)
                    tokens = len(self.text_processor.tokenizer.encode(chunk_text))
                    
                    # Detect chunk type
                    is_heading = self.text_processor.detect_heading(chunk_text)
                    is_conclusion = self.text_processor.detect_conclusion(chunk_text)
                    is_introduction = self.text_processor.detect_introduction(chunk_text)
                    is_reference = self.text_processor.detect_reference(chunk_text)
                    
                    # Calculate quality scores
                    coherence_score = self.text_processor.calculate_coherence(sentences)
                    completeness_score = self.text_processor.calculate_completeness(chunk_text)
                    quality_score = self.quality_optimizer.calculate_chunk_quality(chunk_text)
                    
                    # Determine content type
                    content_type = self._determine_content_type(chunk_text, is_heading, is_reference)
                    
                    # Create metadata
                    metadata = ImprovedChunkMetadata(
                        section_name=f"Section_{i}",
                        chunk_id=f"chunk_{i}",
                        chunk_index=i,
                        tokens=tokens,
                        page_start=page_ranges[0] if page_ranges else 0,
                        page_end=page_ranges[-1] if page_ranges else 0,
                        parent_heading=f"Section_{i}",
                        source_doc_id=doc_id,
                        heading_level=1 if is_heading else 0,
                        content_type=content_type,
                        confidence_score=quality_score,
                        semantic_topic=f"Topic_{i}",
                        word_count=word_count,
                        sentence_count=sentence_count,
                        is_heading=is_heading,
                        is_conclusion=is_conclusion,
                        is_introduction=is_introduction,
                        is_reference=is_reference,
                        is_table=False,
                        is_figure=False,
                        is_formula=False,
                        coherence_score=coherence_score,
                        completeness_score=completeness_score
                    )
                    
                    # Create document chunk
                    document_chunk = ImprovedDocumentChunk(
                        text=chunk_text,
                        metadata=metadata
                    )
                    
                    chunks.append(document_chunk)
                
                return chunks
                
        except Exception:
            return []
    
    def _determine_content_type(self, text: str, is_heading: bool, is_reference: bool) -> str:
        """Determine content type"""
        if is_heading:
            return "heading"
        elif is_reference:
            return "reference"
        elif "table" in text.lower() or "|" in text:
            return "table"
        elif "figure" in text.lower() or "image" in text.lower():
            return "figure"
        elif any(char in text for char in ['∑', '∏', '∫', '∂', '∇', '∆', 'α', 'β', 'γ']):
            return "formula"
        else:
            return "text"
    
    def _optimize_chunks(self, chunks: List[ImprovedDocumentChunk]) -> List[ImprovedDocumentChunk]:
        """Optimize chunks for quality"""
        # Remove low-quality chunks
        quality_chunks = [chunk for chunk in chunks if chunk.metadata.confidence_score > 0.3]
        
        # Sort by quality score
        quality_chunks.sort(key=lambda x: x.metadata.confidence_score, reverse=True)
        
        return quality_chunks
    
    def export_chunks(self, chunks: List[ImprovedDocumentChunk], output_path: str) -> None:
        """Export chunks to JSON"""
        chunk_data = [chunk.to_dict() for chunk in chunks]
        
        output = {
            'filename': 'processed_document.pdf',
            'doc_id': chunks[0].metadata.source_doc_id if chunks else '',
            'total_chunks': len(chunks),
            'chunks': chunk_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    pipeline = ImprovedDocumentChunkingPipeline(
        max_tokens=1000,
        overlap_percentage=0.1
    )
    
    try:
        chunks = pipeline.process_document(
            pdf_path="example.pdf",
            doc_id="doc_001"
        )
        
        pipeline.export_chunks(chunks, "improved_chunks_output.json")
        
        print(f"Successfully processed document into {len(chunks)} chunks")
        
        if chunks:
            print(f"First chunk: {chunks[0].metadata.section_name}")
            print(f"Content type: {chunks[0].metadata.content_type}")
            print(f"Confidence: {chunks[0].metadata.confidence_score:.2f}")
            print(f"Coherence: {chunks[0].metadata.coherence_score:.2f}")
            print(f"Completeness: {chunks[0].metadata.completeness_score:.2f}")
            
    except Exception as e:
        print(f"Error: {e}") 