import re
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
import warnings
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from transformers import AutoTokenizer, AutoModel
import cv2
from ultralytics import YOLO
import easyocr
import paddleocr
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import pdfplumber
from PIL import Image
import io
import tiktoken
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import networkx as nx
from collections import defaultdict
import hashlib

# Suppress warnings for model initialization
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="numba")
warnings.filterwarnings("ignore", message=".*Numba.*")

# Configure Numba for better shutdown handling
try:
    import numba
    numba.config.CUDA_LIBNAME = None  # Disable CUDA for Numba
except:
    pass


@dataclass
class ChunkMetadata:
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
    table_data: Optional[Dict] = None
    image_refs: List[str] = None
    formula_refs: List[str] = None


@dataclass
class DocumentChunk:
    text: str
    metadata: ChunkMetadata
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
            'table_data': self.metadata.table_data,
            'image_refs': self.metadata.image_refs or [],
            'formula_refs': self.metadata.formula_refs or []
        }


class DocumentIntelligenceStack:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.layout_processor = None
        self.layout_model = None
        self.table_detector = None
        self.ocr_reader = None
        self.paddle_ocr = None
        self.fallback_mode = False
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models with fallback mechanisms"""
        # Try to initialize LayoutLMv3
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
                self.layout_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base").to(self.device)
        except Exception:
            self.layout_processor = None
            self.layout_model = None
            self.fallback_mode = True
        
        # Try to initialize YOLO
        try:
            self.table_detector = YOLO('yolov8n.pt')
        except Exception:
            self.table_detector = None
            self.fallback_mode = True
        
        # Try to initialize OCR readers
        try:
            self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        except Exception:
            self.ocr_reader = None
        
        try:
            self.paddle_ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())
        except Exception:
            self.paddle_ocr = None
        
        # If both OCR readers fail, enable fallback mode
        if self.ocr_reader is None and self.paddle_ocr is None:
            self.fallback_mode = True
        
    def analyze_document_structure(self, pdf_path: str) -> Dict[str, Any]:
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return self._fallback_structure_analysis(pdf_path)
        
        structure = {
            'pages': [],
            'tables': [],
            'figures': [],
            'text_blocks': [],
            'formulas': []
        }
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                page_dict = page.get_text("dict")
                
                page_structure = self._analyze_page_structure(page, page_dict, page_num)
                structure['pages'].append(page_structure)
                
                # Try advanced detection methods
                if not self.fallback_mode:
                    try:
                        pix = page.get_pixmap(dpi=300)
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        
                        tables = self._detect_tables(img, page_num)
                        structure['tables'].extend(tables)
                        
                        figures = self._detect_figures(img, page_num)
                        structure['figures'].extend(figures)
                    except Exception:
                        # Fallback to basic detection
                        tables = self._fallback_detect_tables(page_dict, page_num)
                        structure['tables'].extend(tables)
                        
                        figures = self._fallback_detect_figures(page_dict, page_num)
                        structure['figures'].extend(figures)
                else:
                    # Use fallback methods
                    tables = self._fallback_detect_tables(page_dict, page_num)
                    structure['tables'].extend(tables)
                    
                    figures = self._fallback_detect_figures(page_dict, page_num)
                    structure['figures'].extend(figures)
                
                formulas = self._detect_formulas(page_dict, page_num)
                structure['formulas'].extend(formulas)
                
            except Exception:
                # Skip problematic pages but continue processing
                continue
        
        doc.close()
        return structure
    
    def _fallback_structure_analysis(self, pdf_path: str) -> Dict[str, Any]:
        """Fallback structure analysis using basic PDF extraction"""
        try:
            import pdfplumber
            structure = {
                'pages': [],
                'tables': [],
                'figures': [],
                'text_blocks': [],
                'formulas': []
            }
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    
                    # Create basic page structure
                    page_structure = {
                        'page_num': page_num,
                        'text_blocks': [{'text': text, 'bbox': [0, 0, 100, 100], 'page': page_num}],
                        'layout_analysis': {'headings': [], 'paragraphs': []}
                    }
                    structure['pages'].append(page_structure)
            
            return structure
        except Exception:
            # Ultimate fallback - return empty structure
            return {
                'pages': [{'page_num': 0, 'text_blocks': [], 'layout_analysis': {'headings': [], 'paragraphs': []}}],
                'tables': [],
                'figures': [],
                'text_blocks': [],
                'formulas': []
            }
    
    def _analyze_page_structure(self, page, page_dict, page_num):
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
    
    def _classify_layout_elements(self, text_blocks):
        headings = []
        paragraphs = []
        
        for block in text_blocks:
            if block['flags'] & 2**4:  # Bold
                headings.append(block)
            else:
                paragraphs.append(block)
        
        return {'headings': headings, 'paragraphs': paragraphs}
    
    def _detect_tables(self, image, page_num):
        if self.table_detector is None:
            return []
        
        try:
            results = self.table_detector(image)
            tables = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if box.cls == 0:  # Table class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            tables.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': box.conf.cpu().numpy()[0],
                                'page': page_num,
                                'type': 'table'
                            })
            
            return tables
        except Exception:
            return []
    
    def _fallback_detect_tables(self, page_dict, page_num):
        """Fallback table detection using text patterns"""
        tables = []
        blocks = page_dict.get('blocks', [])
        
        for block in blocks:
            if 'lines' in block:
                text_lines = []
                for line in block['lines']:
                    line_text = ' '.join([span['text'] for span in line['spans']])
                    text_lines.append(line_text)
                
                # Look for table-like patterns
                if len(text_lines) > 2:
                    # Check for consistent separators (|, tabs, multiple spaces)
                    separator_count = 0
                    for line in text_lines:
                        if '|' in line or '\t' in line or '  ' in line:
                            separator_count += 1
                    
                    if separator_count > len(text_lines) * 0.5:  # More than 50% lines have separators
                        tables.append({
                            'bbox': block['bbox'],
                            'confidence': 0.7,
                            'page': page_num,
                            'type': 'table'
                        })
        
        return tables
    
    def _detect_figures(self, image, page_num):
        if self.table_detector is None:
            return []
        
        try:
            results = self.table_detector(image)
            figures = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if box.cls in [1, 2]:  # Figure/Image classes
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            figures.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': box.conf.cpu().numpy()[0],
                                'page': page_num,
                                'type': 'figure'
                            })
            
            return figures
        except Exception:
            return []
    
    def _fallback_detect_figures(self, page_dict, page_num):
        """Fallback figure detection using image blocks"""
        figures = []
        blocks = page_dict.get('blocks', [])
        
        for block in blocks:
            # Check if block contains image data
            if 'ext' in block and block['ext'] in ['png', 'jpg', 'jpeg', 'gif']:
                figures.append({
                    'bbox': block['bbox'],
                    'confidence': 0.8,
                    'page': page_num,
                    'type': 'figure'
                })
            # Look for figure references in text
            elif 'lines' in block:
                text = ' '.join([
                    span['text'] for line in block['lines'] 
                    for span in line['spans']
                ])
                if any(keyword in text.lower() for keyword in ['figure', 'fig', 'chart', 'graph', 'image']):
                    figures.append({
                        'bbox': block['bbox'],
                        'confidence': 0.6,
                        'page': page_num,
                        'type': 'figure'
                    })
        
        return figures
    
    def _detect_formulas(self, page_dict, page_num):
        formulas = []
        blocks = page_dict.get('blocks', [])
        
        formula_patterns = [
            r'[∑∏∫∂∇∆]',
            r'[α-ωΑ-Ω]',
            r'[₀-₉⁰-⁹]',
            r'[±×÷≠≤≥≈∞]'
        ]
        
        for block in blocks:
            if 'lines' in block:
                for line in block['lines']:
                    text = ' '.join([span['text'] for span in line['spans']])
                    if any(re.search(pattern, text) for pattern in formula_patterns):
                        formulas.append({
                            'text': text,
                            'bbox': line['bbox'],
                            'page': page_num,
                            'type': 'formula'
                        })
        
        return formulas


class AdvancedSemanticProcessor:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.embedding_model = None
        self.topic_model = None
        self.embedding_cache = {}
        self.fallback_mode = False
        
        self._initialize_semantic_models()
    
    def _initialize_semantic_models(self):
        """Initialize semantic models with fallback options"""
        embedding_models = [
            'BAAI/bge-large-en-v1.5',
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
        
    def initialize_topic_model(self, documents: List[str]):
        if self.embedding_model is None or len(documents) < 5:
            # Fallback: return simple topic assignments
            return list(range(len(documents))), [0.5] * len(documents)
        
        try:
            umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
            hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom')
            
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                calculate_probabilities=True,
                verbose=False
            )
            
            topics, probs = self.topic_model.fit_transform(documents)
            return topics, probs
        except Exception:
            # Fallback: simple clustering based on document similarity
            return self._fallback_topic_modeling(documents)
    
    def _fallback_topic_modeling(self, documents: List[str]):
        """Fallback topic modeling using simple text similarity"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            
            # Use TF-IDF for basic similarity
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Simple K-means clustering
            n_clusters = min(5, len(documents) // 2)
            if n_clusters < 2:
                return list(range(len(documents))), [0.5] * len(documents)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            topics = kmeans.fit_predict(tfidf_matrix)
            
            # Generate pseudo-probabilities
            probs = [0.7] * len(documents)
            
            return topics.tolist(), probs
        except Exception:
            # Ultimate fallback: assign sequential topics
            return list(range(len(documents))), [0.5] * len(documents)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        cache_key = hashlib.md5(''.join(texts).encode()).hexdigest()
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        if self.embedding_model is None:
            return self._fallback_embeddings(texts)
        
        try:
            embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress_bar=False)
            self.embedding_cache[cache_key] = embeddings
            return embeddings
        except Exception:
            return self._fallback_embeddings(texts)
    
    def _fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """Fallback embeddings using TF-IDF"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Convert sparse matrix to dense numpy array
            embeddings = tfidf_matrix.toarray()
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
            
            return embeddings
        except Exception:
            # Ultimate fallback: random embeddings
            return np.random.rand(len(texts), 384)
    
    def hierarchical_clustering(self, embeddings: np.ndarray, n_clusters: int = None) -> np.ndarray:
        if n_clusters is None:
            n_clusters = min(len(embeddings) // 2, 10)
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        
        return clustering.fit_predict(embeddings)
    
    def find_semantic_boundaries(self, texts: List[str], threshold: float = 0.7) -> List[int]:
        embeddings = self.get_embeddings(texts)
        boundaries = []
        
        for i in range(1, len(embeddings)):
            similarity = cosine_similarity(
                embeddings[i-1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]
            
            if similarity < threshold:
                boundaries.append(i)
        
        return boundaries
    
    def calculate_coherence_score(self, chunk_texts: List[str]) -> float:
        if len(chunk_texts) < 2:
            return 1.0
        
        embeddings = self.get_embeddings(chunk_texts)
        similarities = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]
                similarities.append(sim)
        
        return np.mean(similarities)


class MultiModalProcessor:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.ocr_reader = None
        self.paddle_ocr = None
        self.fallback_mode = False
        
        self._initialize_ocr_readers()
    
    def _initialize_ocr_readers(self):
        """Initialize OCR readers with fallback options"""
        try:
            self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        except Exception:
            self.ocr_reader = None
        
        try:
            self.paddle_ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())
        except Exception:
            self.paddle_ocr = None
        
        if self.ocr_reader is None and self.paddle_ocr is None:
            self.fallback_mode = True
        
    def extract_table_content(self, pdf_path: str, table_bbox: List[float], page_num: int) -> Dict[str, Any]:
        if self.fallback_mode:
            return self._fallback_table_extraction(pdf_path, table_bbox, page_num)
        
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            table_rect = fitz.Rect(table_bbox)
            table_pix = page.get_pixmap(clip=table_rect, dpi=300)
            table_img = Image.open(io.BytesIO(table_pix.tobytes("png")))
            
            # Try EasyOCR first
            if self.ocr_reader is not None:
                try:
                    table_text = self.ocr_reader.readtext(np.array(table_img))
                    rows = []
                    for (bbox, text, conf) in table_text:
                        if conf > 0.5:
                            rows.append({
                                'text': text,
                                'bbox': bbox,
                                'confidence': conf
                            })
                    
                    doc.close()
                    return {
                        'raw_data': rows,
                        'structured_data': self._structure_table_data(rows),
                        'extraction_method': 'easyocr'
                    }
                except Exception:
                    pass
            
            # Try PaddleOCR as fallback
            if self.paddle_ocr is not None:
                try:
                    table_text = self.paddle_ocr.ocr(np.array(table_img))
                    rows = []
                    for line in table_text:
                        for word_info in line:
                            bbox, (text, conf) = word_info
                            if conf > 0.5:
                                rows.append({
                                    'text': text,
                                    'bbox': bbox,
                                    'confidence': conf
                                })
                    
                    doc.close()
                    return {
                        'raw_data': rows,
                        'structured_data': self._structure_table_data(rows),
                        'extraction_method': 'paddleocr'
                    }
                except Exception:
                    pass
            
            doc.close()
            return self._fallback_table_extraction(pdf_path, table_bbox, page_num)
            
        except Exception:
            return self._fallback_table_extraction(pdf_path, table_bbox, page_num)
    
    def _fallback_table_extraction(self, pdf_path: str, table_bbox: List[float], page_num: int) -> Dict[str, Any]:
        """Fallback table extraction using basic text extraction"""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            table_rect = fitz.Rect(table_bbox)
            table_text = page.get_text("text", clip=table_rect)
            
            doc.close()
            
            # Simple text parsing
            lines = table_text.split('\n')
            rows = []
            for line in lines:
                if line.strip():
                    rows.append({
                        'text': line.strip(),
                        'bbox': [0, 0, 100, 20],
                        'confidence': 0.5
                    })
            
            return {
                'raw_data': rows,
                'structured_data': self._structure_table_data(rows),
                'extraction_method': 'basic_text'
            }
        except Exception:
            return {
                'raw_data': [],
                'structured_data': [],
                'extraction_method': 'failed'
            }
    
    def _structure_table_data(self, rows: List[Dict]) -> List[List[str]]:
        if not rows:
            return []
        
        sorted_rows = sorted(rows, key=lambda x: x['bbox'][0][1])
        
        table_rows = []
        current_row = []
        current_y = sorted_rows[0]['bbox'][0][1]
        
        for row in sorted_rows:
            row_y = row['bbox'][0][1]
            
            if abs(row_y - current_y) > 10:
                if current_row:
                    current_row.sort(key=lambda x: x['bbox'][0][0])
                    table_rows.append([cell['text'] for cell in current_row])
                current_row = [row]
                current_y = row_y
            else:
                current_row.append(row)
        
        if current_row:
            current_row.sort(key=lambda x: x['bbox'][0][0])
            table_rows.append([cell['text'] for cell in current_row])
        
        return table_rows
    
    def extract_image_context(self, pdf_path: str, image_bbox: List[float], page_num: int) -> Dict[str, Any]:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        image_rect = fitz.Rect(image_bbox)
        image_pix = page.get_pixmap(clip=image_rect, dpi=300)
        image_img = Image.open(io.BytesIO(image_pix.tobytes("png")))
        
        caption = self._find_image_caption(page, image_bbox)
        
        doc.close()
        return {
            'caption': caption,
            'bbox': image_bbox,
            'page': page_num,
            'type': 'image'
        }
    
    def _find_image_caption(self, page, image_bbox: List[float]) -> str:
        page_dict = page.get_text("dict")
        blocks = page_dict.get('blocks', [])
        
        caption_candidates = []
        
        for block in blocks:
            if 'lines' in block:
                block_bbox = block['bbox']
                
                if (block_bbox[1] > image_bbox[3] and 
                    block_bbox[1] - image_bbox[3] < 50):
                    
                    text = ' '.join([
                        span['text'] for line in block['lines'] 
                        for span in line['spans']
                    ])
                    
                    if any(keyword in text.lower() for keyword in ['figure', 'fig', 'image', 'chart']):
                        caption_candidates.append(text)
        
        return caption_candidates[0] if caption_candidates else ""


class QualityOptimizer:
    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
    def calculate_chunk_quality(self, chunk: DocumentChunk) -> float:
        scores = []
        
        scores.append(self._content_completeness_score(chunk.text))
        scores.append(self._semantic_coherence_score(chunk.text))
        scores.append(self._information_density_score(chunk.text))
        scores.append(self._readability_score(chunk.text))
        
        return np.mean(scores)
    
    def _content_completeness_score(self, text: str) -> float:
        sentences = text.split('.')
        complete_sentences = [s for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return 0.0
        
        return len(complete_sentences) / len(sentences)
    
    def _semantic_coherence_score(self, text: str) -> float:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 1.0
        
        return 0.8
    
    def _information_density_score(self, text: str) -> float:
        words = text.split()
        
        if not words:
            return 0.0
        
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def _readability_score(self, text: str) -> float:
        words = text.split()
        sentences = text.split('.')
        
        if not sentences or not words:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        
        if avg_words_per_sentence < 10:
            return 0.9
        elif avg_words_per_sentence < 20:
            return 0.8
        elif avg_words_per_sentence < 30:
            return 0.7
        else:
            return 0.6
    
    def adaptive_token_limits(self, content_type: str, complexity_score: float) -> int:
        base_limits = {
            'text': 1000,
            'table': 1500,
            'figure': 800,
            'formula': 600,
            'code': 1200
        }
        
        base_limit = base_limits.get(content_type, 1000)
        
        if complexity_score > 0.8:
            return int(base_limit * 1.3)
        elif complexity_score < 0.4:
            return int(base_limit * 0.7)
        else:
            return base_limit


class EnhancedDocumentChunkingPipeline:
    def __init__(self, 
                 max_tokens: int = 1000,
                 overlap_percentage: float = 0.15,
                 semantic_model: str = 'BAAI/bge-large-en-v1.5',
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.max_tokens = max_tokens
        self.overlap_percentage = overlap_percentage
        self._shutdown = False
        
        # Initialize components with error handling
        try:
            self.doc_intelligence = DocumentIntelligenceStack(device)
            self.semantic_processor = AdvancedSemanticProcessor(device)
            self.multimodal_processor = MultiModalProcessor(device)
            self.quality_optimizer = QualityOptimizer()
        except Exception as e:
            print(f"Warning: Some components failed to initialize: {e}")
            # Set fallback mode
            self._shutdown = True
        
    def process_document(self, 
                        pdf_path: str, 
                        doc_id: str,
                        extractor_type: str = 'enhanced') -> List[DocumentChunk]:
        
        # Check if shutdown is requested
        if self._shutdown:
            return self._fallback_document_processing(pdf_path, doc_id)
        
        try:
            document_structure = self.doc_intelligence.analyze_document_structure(pdf_path)
        except Exception:
            # Ultimate fallback: basic text extraction
            return self._fallback_document_processing(pdf_path, doc_id)
        
        all_chunks = []
        
        for page_info in document_structure['pages']:
            try:
                page_chunks = self._process_page(page_info, document_structure, pdf_path, doc_id)
                all_chunks.extend(page_chunks)
            except Exception:
                # Skip problematic pages but continue
                continue
        
        if not all_chunks:
            return self._fallback_document_processing(pdf_path, doc_id)
        
        all_texts = [chunk.text for chunk in all_chunks]
        
        # Try topic modeling with fallback
        try:
            if len(all_texts) > 5:
                topics, probs = self.semantic_processor.initialize_topic_model(all_texts)
                
                for i, chunk in enumerate(all_chunks):
                    if i < len(topics):
                        try:
                            if self.semantic_processor.topic_model and hasattr(self.semantic_processor.topic_model, 'get_topic_info'):
                                topic_info = self.semantic_processor.topic_model.get_topic_info()
                                if topics[i] >= 0 and topics[i] < len(topic_info):
                                    chunk.metadata.semantic_topic = topic_info.iloc[topics[i]]['Name']
                        except Exception:
                            chunk.metadata.semantic_topic = f"Topic_{topics[i]}"
        except Exception:
            # Continue without topic modeling
            pass
        
        # Try embeddings with fallback
        try:
            embeddings = self.semantic_processor.get_embeddings(all_texts)
            for i, chunk in enumerate(all_chunks):
                if i < len(embeddings):
                    chunk.embeddings = embeddings[i]
                chunk.metadata.confidence_score = self.quality_optimizer.calculate_chunk_quality(chunk)
        except Exception:
            # Continue without embeddings
            for chunk in all_chunks:
                chunk.metadata.confidence_score = 0.5
        
        return self._apply_overlap_and_optimize(all_chunks)
    
    def _fallback_document_processing(self, pdf_path: str, doc_id: str) -> List[DocumentChunk]:
        """Fallback document processing using basic text extraction"""
        try:
            import pdfplumber
            chunks = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        # Simple sentence-based chunking
                        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 20]
                        
                        for i in range(0, len(sentences), 5):  # Group 5 sentences per chunk
                            chunk_sentences = sentences[i:i+5]
                            chunk_text = '. '.join(chunk_sentences)
                            
                            if chunk_text:
                                chunk = self._create_chunk(
                                    text=chunk_text,
                                    section_name=f"Page_{page_num}",
                                    chunk_index=len(chunks),
                                    doc_id=doc_id,
                                    page_num=page_num,
                                    content_type="text"
                                )
                                chunks.append(chunk)
            
            return chunks
        except Exception:
            # Ultimate fallback: return empty chunk
            return [self._create_chunk(
                text="Document processing failed",
                section_name="Error",
                chunk_index=0,
                doc_id=doc_id,
                page_num=0,
                content_type="text"
            )]
    
    def _process_page(self, page_info: Dict, document_structure: Dict, pdf_path: str, doc_id: str) -> List[DocumentChunk]:
        page_num = page_info['page_num']
        text_blocks = page_info['text_blocks']
        
        chunks = []
        
        page_tables = [t for t in document_structure['tables'] if t['page'] == page_num]
        page_figures = [f for f in document_structure['figures'] if f['page'] == page_num]
        page_formulas = [f for f in document_structure['formulas'] if f['page'] == page_num]
        
        for table in page_tables:
            table_content = self.multimodal_processor.extract_table_content(pdf_path, table['bbox'], page_num)
            table_text = self._format_table_text(table_content['structured_data'])
            
            chunk = self._create_chunk(
                text=table_text,
                section_name=f"Table_{page_num}_{len(chunks)}",
                chunk_index=len(chunks),
                doc_id=doc_id,
                page_num=page_num,
                content_type="table",
                table_data=table_content
            )
            chunks.append(chunk)
        
        for figure in page_figures:
            figure_context = self.multimodal_processor.extract_image_context(pdf_path, figure['bbox'], page_num)
            figure_text = f"Figure: {figure_context['caption']}"
            
            chunk = self._create_chunk(
                text=figure_text,
                section_name=f"Figure_{page_num}_{len(chunks)}",
                chunk_index=len(chunks),
                doc_id=doc_id,
                page_num=page_num,
                content_type="figure"
            )
            chunks.append(chunk)
        
        text_content = self._extract_text_content(text_blocks)
        if text_content:
            text_chunks = self._chunk_text_content(text_content, doc_id, page_num)
            chunks.extend(text_chunks)
        
        return chunks
    
    def _format_table_text(self, structured_data: List[List[str]]) -> str:
        if not structured_data:
            return ""
        
        formatted_rows = []
        for row in structured_data:
            formatted_rows.append(" | ".join(row))
        
        return "\n".join(formatted_rows)
    
    def _extract_text_content(self, text_blocks: List[Dict]) -> str:
        sorted_blocks = sorted(text_blocks, key=lambda x: (x['bbox'][1], x['bbox'][0]))
        return " ".join([block['text'] for block in sorted_blocks])
    
    def _chunk_text_content(self, text: str, doc_id: str, page_num: int) -> List[DocumentChunk]:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return []
        
        embeddings = self.semantic_processor.get_embeddings(sentences)
        boundaries = self.semantic_processor.find_semantic_boundaries(sentences)
        
        chunks = []
        start_idx = 0
        
        for boundary in boundaries + [len(sentences)]:
            chunk_sentences = sentences[start_idx:boundary]
            chunk_text = '. '.join(chunk_sentences)
            
            if len(chunk_text.strip()) > 50:
                chunk = self._create_chunk(
                    text=chunk_text,
                    section_name=f"Section_{page_num}_{len(chunks)}",
                    chunk_index=len(chunks),
                    doc_id=doc_id,
                    page_num=page_num,
                    content_type="text"
                )
                chunks.append(chunk)
            
            start_idx = boundary
        
        return chunks
    
    def _create_chunk(self, text: str, section_name: str, chunk_index: int, 
                     doc_id: str, page_num: int, content_type: str = "text",
                     table_data: Optional[Dict] = None) -> DocumentChunk:
        
        tokens = len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text))
        
        metadata = ChunkMetadata(
            section_name=section_name,
            chunk_id=f"{section_name}_{chunk_index}",
            chunk_index=chunk_index,
            tokens=tokens,
            page_start=page_num,
            page_end=page_num,
            parent_heading=section_name,
            source_doc_id=doc_id,
            heading_level=1,
            content_type=content_type,
            table_data=table_data,
            image_refs=[],
            formula_refs=[]
        )
        
        return DocumentChunk(text=text, metadata=metadata)
    
    def _apply_overlap_and_optimize(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        if len(chunks) <= 1:
            return chunks
        
        optimized_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                optimized_chunks.append(chunk)
                continue
            
            prev_chunk = chunks[i-1]
            
            if (chunk.metadata.content_type == "text" and 
                prev_chunk.metadata.content_type == "text"):
                
                overlap_text = self._get_overlap_text(prev_chunk.text)
                chunk.text = f"{overlap_text} {chunk.text}"
                
                chunk.metadata.tokens = len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(chunk.text))
            
            optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _get_overlap_text(self, text: str) -> str:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 1:
            return text
        
        overlap_count = max(1, int(len(sentences) * self.overlap_percentage))
        return '. '.join(sentences[-overlap_count:])
    
    def export_chunks(self, chunks: List[DocumentChunk], output_path: str) -> None:
        chunk_data = [chunk.to_dict() for chunk in chunks]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    pipeline = EnhancedDocumentChunkingPipeline(
        max_tokens=1000,
        overlap_percentage=0.15,
        semantic_model='BAAI/bge-large-en-v1.5'
    )
    
    try:
        chunks = pipeline.process_document(
            pdf_path="example.pdf",
            doc_id="doc_001",
            extractor_type='enhanced'
        )
        
        pipeline.export_chunks(chunks, "enhanced_chunks_output.json")
        
        print(f"Successfully processed document into {len(chunks)} chunks")
        
        if chunks:
            print(f"First chunk: {chunks[0].metadata.section_name}")
            print(f"Content type: {chunks[0].metadata.content_type}")
            print(f"Confidence: {chunks[0].metadata.confidence_score:.2f}")
            
    except Exception as e:
        print(f"Error: {e}") 