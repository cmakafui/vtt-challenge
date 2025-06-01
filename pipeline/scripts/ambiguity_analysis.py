#!/usr/bin/env python3
"""
VTT Innovation Ambiguity Analysis with Azure OpenAI
Uses all metadata including URLs, titles, dates, and document types
"""

import pickle
import hashlib
import json
import time
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich import box

# Azure OpenAI imports
from openai import AzureOpenAI

# FAISS imports with fallback
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not installed. Install with: pip install faiss-cpu")


# Pydantic Models
class InnovationMention(BaseModel):
    """Innovation mention with full context from original data"""

    unique_mention_id: str
    original_id: str
    name: str
    description: str = ""
    source_doc_id: str
    dataset_origin: str
    associated_orgs: List[str] = Field(default_factory=list)
    non_vtt_orgs: List[str] = Field(default_factory=list)  # Organizations excluding VTT

    # Original context from CSV data
    source_url: Optional[str] = None
    page_title: Optional[str] = None
    website_domain: Optional[str] = None
    publication_date: Optional[str] = None
    document_type: Optional[str] = None
    company_name: Optional[str] = None
    has_vtt_involvement: bool = False


class DuplicateMatch(BaseModel):
    """Duplicate match with context-based reasoning"""

    innovation1: InnovationMention
    innovation2: InnovationMention
    similarity_score: float
    shared_non_vtt_orgs: List[str]
    confidence_level: str = "medium"
    match_reasons: List[str] = Field(default_factory=list)
    url_match: bool = False
    domain_match: bool = False


class AnalysisResults(BaseModel):
    """Complete analysis results"""

    total_raw_mentions: int
    unique_innovations: int
    duplicates: List[DuplicateMatch]
    source_counts: Dict[str, int]
    processing_time: float
    config: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class EmbeddingCache(BaseModel):
    """Cache structure for embeddings"""

    model: str
    embeddings: Dict[str, List[float]]
    metadata: Dict[str, str]
    created_at: datetime = Field(default_factory=datetime.now)


app = typer.Typer(
    name="vtt-analysis-azure",
    help="🚀 VTT Innovation Analysis with Azure OpenAI",
    add_completion=False,
)
console = Console()

# VTT organization ID for identification (not filtering)
VTT_ORG_ID = "FI26473754"


class AzureOpenAIEmbeddings:
    def __init__(
        self,
        config_path: str = "data/keys/azure_config.json",
        model_key: str = "gpt-4o-mini",
    ):
        """
        Initialize Azure OpenAI client for embeddings using your config file.

        Args:
            config_path: Path to your Azure config JSON file
            model_key: Which model configuration to use from your config
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.model_config = self.config[model_key]

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.model_config["api_key"],
            api_version=self.model_config["api_version"],
            azure_endpoint=self.model_config["api_base"],
        )

        # Use the embedding deployment name
        self.embedding_deployment = self.model_config["emb_deployment"]
        # Set default dimensions for text-embedding-3-large
        self.dimensions = 1024

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text string.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_deployment, input=text, dimensions=self.dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            console.print(f"[red]Error getting embedding: {e}[/red]")
            return None

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in a single API call.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        try:
            # Azure OpenAI supports up to 2,048 input items per request
            if len(texts) > 2048:
                console.print(
                    f"[yellow]Warning: Input has {len(texts)} items, but max is 2048. Processing in batches...[/yellow]"
                )
                return self._process_large_batch(texts)

            response = self.client.embeddings.create(
                model=self.embedding_deployment, input=texts, dimensions=self.dimensions
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            console.print(f"[red]Error getting batch embeddings: {e}[/red]")
            return None

    def _process_large_batch(
        self, texts: List[str], batch_size: int = 1024
    ) -> List[List[float]]:
        """Process large batches by splitting into smaller chunks."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.get_embeddings_batch(batch)
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1, vec2: Embedding vectors

        Returns:
            Cosine similarity score between -1 and 1
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        return dot_product / (norm1 * norm2)


class EmbeddingCacheManager:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings_cache.json"
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, List[float]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    cache_data = json.load(f)
                    return cache_data.get("embeddings", {})
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load cache: {e}[/yellow]")
        return {}

    def _save_cache(self):
        try:
            cache_obj = EmbeddingCache(
                model=getattr(self, "_current_model", "unknown"),
                embeddings=self._cache,
                metadata={"total_embeddings": str(len(self._cache))},
            )
            with open(self.cache_file, "w") as f:
                json.dump(cache_obj.model_dump(), f, indent=2, default=str)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save cache: {e}[/yellow]")

    def get_cache_key(self, text: str, model: str) -> str:
        return hashlib.md5(f"{text}:{model}".encode()).hexdigest()

    def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        key = self.get_cache_key(text, model)
        return self._cache.get(key)

    def store_embedding(self, text: str, model: str, embedding: List[float]):
        key = self.get_cache_key(text, model)
        self._cache[key] = embedding
        self._current_model = model

    def save_to_disk(self):
        self._save_cache()


class FAISSSearchEngine:
    """FAISS-based similarity search engine"""

    def __init__(self):
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required but not installed. Install with: pip install faiss-cpu"
            )

        self.index = None
        self.id_map = []
        self.innovation_map = {}
        self.dimension = None

    def build_index(
        self, innovations: List[InnovationMention], embeddings: Dict[str, List[float]]
    ) -> float:
        """Build FAISS index from embeddings"""
        start_time = time.time()

        valid_innovations = [
            inn for inn in innovations if inn.unique_mention_id in embeddings
        ]

        if not valid_innovations:
            raise ValueError("No valid embeddings found for building FAISS index")

        sample_embedding = embeddings[valid_innovations[0].unique_mention_id]
        self.dimension = len(sample_embedding)

        self.index = faiss.IndexFlatIP(self.dimension)

        embedding_matrix = np.array(
            [embeddings[inn.unique_mention_id] for inn in valid_innovations],
            dtype=np.float32,
        )

        faiss.normalize_L2(embedding_matrix)
        self.index.add(embedding_matrix)

        self.id_map = [inn.unique_mention_id for inn in valid_innovations]
        self.innovation_map = {inn.unique_mention_id: inn for inn in valid_innovations}

        build_time = time.time() - start_time
        console.print(
            f"[green]✅ FAISS index built: {len(valid_innovations)} vectors, {build_time:.2f}s[/green]"
        )

        return build_time

    def search_similar(
        self, innovation_id: str, embeddings: Dict[str, List[float]], k: int = 100
    ) -> List[Tuple[str, float]]:
        """Search for similar innovations using FAISS"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        if innovation_id not in embeddings:
            return []

        query_embedding = np.array([embeddings[innovation_id]], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        similarities, indices = self.index.search(query_embedding, k + 1)

        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue

            candidate_id = self.id_map[idx]
            if candidate_id == innovation_id:
                continue

            results.append((candidate_id, float(similarity)))

        return results


def load_original_context_data(
    comp_csv_path: str, vtt_csv_path: str
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Load original CSV data to preserve all metadata"""

    comp_context = {}
    vtt_context = {}

    # Load company domain data
    if Path(comp_csv_path).exists():
        df_comp = pd.read_csv(comp_csv_path)
        df_comp = df_comp[df_comp["Website"].str.startswith("www.", na=False)]
        df_comp["source_index"] = df_comp.index

        for _, row in df_comp.iterrows():
            key = f"{row['Company name'].replace(' ', '_')}_{row['source_index']}"
            comp_context[key] = {
                "source_url": row["Link"],
                "page_title": row["Title"],
                "website_domain": row["Website"],
                "publication_date": row["date_obtained"],
                "document_type": row["Type"],
                "company_name": row["Company name"],
                "orbis_id": row["Orbis ID"],
            }

    # Load VTT domain data
    if Path(vtt_csv_path).exists():
        df_vtt = pd.read_csv(vtt_csv_path)

        for idx, row in df_vtt.iterrows():
            key = f"{row['Vat_id'].replace(' ', '_')}_{idx}"
            vtt_context[key] = {
                "source_url": row["source_url"],
                "page_title": row.get("title", "VTT Domain Page"),
                "website_domain": "vtt_domain",
                "publication_date": row.get("date_obtained", "Unknown"),
                "document_type": "VTT_Domain",
                "company_name": "VTT",
                "vat_id": row["Vat_id"],
            }

    console.print(
        f"[cyan]📊 Loaded context for {len(comp_context)} company docs, {len(vtt_context)} VTT docs[/cyan]"
    )
    return comp_context, vtt_context


def setup_azure_openai_client(
    config_path: str, model_key: str
) -> AzureOpenAIEmbeddings:
    """Setup and test Azure OpenAI client connection"""
    console.print(f"[cyan]🤖 Connecting to Azure OpenAI...[/cyan]")
    try:
        embeddings_client = AzureOpenAIEmbeddings(
            config_path=config_path, model_key=model_key
        )

        # Test connection with a simple embedding
        test_embedding = embeddings_client.get_embedding("test connection")
        if test_embedding is None:
            raise Exception("Failed to get test embedding")

        console.print(
            f"[green]✅ Connected to Azure OpenAI. Using deployment '{embeddings_client.embedding_deployment}'.[/green]"
        )
        return embeddings_client
    except Exception as e:
        console.print(f"[red]❌ Failed to connect to Azure OpenAI: {e}[/red]")
        raise typer.Exit(1)


def load_and_validate_data(data_paths: List[Path]) -> Tuple[List[Any], Dict[str, int]]:
    """Load and validate data from multiple directories"""
    all_graph_docs = []
    source_counts = defaultdict(int)
    langchain_errors = 0

    for directory_idx, directory in enumerate(data_paths):
        pkl_files = list(directory.glob("*.pkl"))
        if not pkl_files:
            console.print(
                f"[yellow]Warning: No .pkl files found in {directory}. Skipping.[/yellow]"
            )
            continue

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            transient=True,
        ) as progress:
            task_desc = f"Loading from {directory.name} ({directory_idx + 1}/{len(data_paths)})..."
            task = progress.add_task(task_desc, total=len(pkl_files))

            for filepath in pkl_files:
                try:
                    with open(filepath, "rb") as f:
                        loaded_obj = pickle.load(f)
                        doc = (
                            loaded_obj[0]
                            if isinstance(loaded_obj, list) and len(loaded_obj) > 0
                            else loaded_obj
                        )

                        if doc.source_document is not None:
                            if doc.source_document.metadata is None:
                                doc.source_document.metadata = {}
                            doc.source_document.metadata["dataset_origin"] = (
                                directory.name
                            )

                        all_graph_docs.append(doc)
                        source_counts[directory.name] += 1
                    progress.advance(task)
                except ModuleNotFoundError as e:
                    if "langchain" in str(e):
                        langchain_errors += 1
                    progress.advance(task)
                except Exception as e:
                    console.print(f"[red]Error loading {filepath.name}: {e}[/red]")
                    progress.advance(task)

    if langchain_errors > 0:
        console.print(
            Panel(
                f"[red]⚠️ Failed to load {langchain_errors} files due to missing LangChain modules.[/red]\n"
                "[yellow]Fix: pip install langchain-core langchain[/yellow]",
                title="🔧 Dependency Issue",
                style="red",
            )
        )

    return all_graph_docs, dict(source_counts)


def extract_innovations_with_context(
    graph_docs: List[Any], comp_context: Dict[str, Dict], vtt_context: Dict[str, Dict]
) -> List[InnovationMention]:
    """Extract innovation mentions with full original context"""
    innovations = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            "Extracting innovations with context...", total=len(graph_docs)
        )

        for doc_idx, doc in enumerate(graph_docs):
            # Try to find original context for this document
            doc_context = None
            source_doc_id = f"unknown_doc_{doc_idx}"

            if doc.source_document and doc.source_document.metadata:
                source_id = doc.source_document.metadata.get("source_id", "")

                # Try both context dictionaries
                if source_id in comp_context:
                    doc_context = comp_context[source_id]
                    source_doc_id = source_id
                elif source_id in vtt_context:
                    doc_context = vtt_context[source_id]
                    source_doc_id = source_id

            innovations_in_doc = [n for n in doc.nodes if n.type == "Innovation"]

            for innov_idx, innovation in enumerate(innovations_in_doc):
                unique_mention_id = f"{innovation.id}_{doc_idx}_{innov_idx}"

                # Extract all associated organizations
                associated_orgs = []
                for rel in doc.relationships:
                    if rel.source == innovation.id or rel.target == innovation.id:
                        org_id_in_rel = (
                            rel.target if rel.source == innovation.id else rel.source
                        )
                        org_node = next(
                            (n for n in doc.nodes if n.id == org_id_in_rel), None
                        )
                        if org_node and org_node.type == "Organization":
                            associated_orgs.append(org_id_in_rel)

                # Separate VTT from non-VTT organizations
                non_vtt_orgs = [org for org in associated_orgs if org != VTT_ORG_ID]
                has_vtt = VTT_ORG_ID in associated_orgs

                dataset_origin = "unknown_source"
                if doc.source_document and doc.source_document.metadata:
                    dataset_origin = doc.source_document.metadata.get(
                        "dataset_origin", "unknown_source"
                    )

                innovation_mention = InnovationMention(
                    unique_mention_id=unique_mention_id,
                    original_id=innovation.id,
                    name=innovation.properties.get("english_id", innovation.id),
                    description=innovation.properties.get("description", ""),
                    source_doc_id=source_doc_id,
                    dataset_origin=dataset_origin,
                    associated_orgs=list(set(associated_orgs)),
                    non_vtt_orgs=non_vtt_orgs,
                    has_vtt_involvement=has_vtt,
                    # Add original context if available
                    source_url=doc_context["source_url"] if doc_context else None,
                    page_title=doc_context["page_title"] if doc_context else None,
                    website_domain=doc_context["website_domain"]
                    if doc_context
                    else None,
                    publication_date=doc_context["publication_date"]
                    if doc_context
                    else None,
                    document_type=doc_context["document_type"] if doc_context else None,
                    company_name=doc_context["company_name"] if doc_context else None,
                )
                innovations.append(innovation_mention)
            progress.advance(task)

    return innovations


def generate_embeddings_batch(
    innovations: List[InnovationMention],
    azure_client: AzureOpenAIEmbeddings,
    batch_size: int = 50,
    dimensions: int = 1024,
    cache_manager: Optional[EmbeddingCacheManager] = None,
) -> Dict[str, List[float]]:
    """Generate embeddings in batches with caching support"""
    embeddings = {}
    texts_to_embed = []

    # Check cache first
    if cache_manager:
        for innovation in innovations:
            # Include more context in embedding text
            text_parts = [innovation.name, innovation.description]
            if innovation.page_title and innovation.page_title != innovation.name:
                text_parts.append(innovation.page_title)
            text = " ".join(filter(None, text_parts))

            cached_embedding = cache_manager.get_embedding(
                text, azure_client.embedding_deployment
            )
            if cached_embedding:
                embeddings[innovation.unique_mention_id] = cached_embedding
            else:
                texts_to_embed.append((innovation.unique_mention_id, text))
    else:
        texts_to_embed = []
        for inn in innovations:
            text_parts = [inn.name, inn.description]
            if inn.page_title and inn.page_title != inn.name:
                text_parts.append(inn.page_title)
            text = " ".join(filter(None, text_parts))
            texts_to_embed.append((inn.unique_mention_id, text))

    if not texts_to_embed:
        console.print("[green]✅ All embeddings found in cache![/green]")
        return embeddings

    console.print(
        f"[cyan]Generating {len(texts_to_embed)} new embeddings (batch size: {batch_size}, dimensions: {dimensions})...[/cyan]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=True,
    ) as progress:
        total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size
        task = progress.add_task("Generating embeddings...", total=total_batches)

        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = [text for _, text in texts_to_embed[i : i + batch_size]]
            batch_ids = [uid for uid, _ in texts_to_embed[i : i + batch_size]]

            try:
                batch_embeddings = azure_client.get_embeddings_batch(batch_texts)

                if batch_embeddings:
                    for j, embedding in enumerate(batch_embeddings):
                        if j < len(batch_ids) and embedding:
                            embeddings[batch_ids[j]] = embedding

                            if cache_manager and j < len(batch_texts):
                                cache_manager.store_embedding(
                                    batch_texts[j],
                                    azure_client.embedding_deployment,
                                    embedding,
                                )

            except Exception as e:
                console.print(f"[red]Error in batch embedding: {e}[/red]")

            progress.advance(task)

    if cache_manager:
        cache_manager.save_to_disk()

    return embeddings


def find_duplicates_with_context(
    innovations: List[InnovationMention],
    embeddings: Dict[str, List[float]],
    faiss_engine: FAISSSearchEngine,
    similarity_threshold: float = 0.80,
    min_shared_orgs: int = 1,
    top_k_candidates: int = 100,
) -> List[DuplicateMatch]:
    """Duplicate detection using full context information"""
    duplicates = []
    seen_pairs = set()

    valid_innovations = [
        inn for inn in innovations if inn.unique_mention_id in embeddings
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            "Finding duplicates with context...", total=len(valid_innovations)
        )

        for innovation in valid_innovations:
            similar_candidates = faiss_engine.search_similar(
                innovation.unique_mention_id, embeddings, k=top_k_candidates
            )

            for candidate_id, similarity in similar_candidates:
                if similarity >= similarity_threshold:
                    candidate = faiss_engine.innovation_map.get(candidate_id)
                    if not candidate:
                        continue

                    # Check shared non-VTT organizations
                    org_overlap = set(innovation.non_vtt_orgs) & set(
                        candidate.non_vtt_orgs
                    )

                    # Only consider if they share non-VTT organizations OR have strong contextual signals
                    url_match = bool(
                        innovation.source_url
                        and candidate.source_url
                        and innovation.source_url == candidate.source_url
                    )
                    domain_match = bool(
                        innovation.website_domain
                        and candidate.website_domain
                        and innovation.website_domain == candidate.website_domain
                    )

                    if len(org_overlap) >= min_shared_orgs or url_match:
                        confidence_boost = 0.0
                        match_reasons = []

                        # URL-based signals (strongest)
                        if url_match:
                            confidence_boost += 0.15
                            match_reasons.append("Same source URL")
                        elif domain_match:
                            confidence_boost += 0.08
                            match_reasons.append("Same website domain")

                        # Organizational overlap
                        if len(org_overlap) > 0:
                            match_reasons.append(
                                f"Shared organizations: {len(org_overlap)}"
                            )
                            if len(org_overlap) > 1:
                                confidence_boost += 0.05

                        # Document type consistency
                        if (
                            innovation.document_type
                            and candidate.document_type
                            and innovation.document_type == candidate.document_type
                        ):
                            confidence_boost += 0.03
                            match_reasons.append("Same document type")

                        # Company context (for partner domain)
                        if (
                            innovation.company_name
                            and candidate.company_name
                            and innovation.company_name == candidate.company_name
                        ):
                            confidence_boost += 0.05
                            match_reasons.append("Same company source")

                        # Adjust final similarity score
                        adjusted_similarity = min(similarity + confidence_boost, 1.0)

                        # Avoid duplicate pairs
                        pair_key = tuple(
                            sorted([innovation.unique_mention_id, candidate_id])
                        )
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)

                            confidence = (
                                "high"
                                if adjusted_similarity >= 0.92 or url_match
                                else "medium"
                                if adjusted_similarity >= 0.85
                                else "low"
                            )

                            duplicate = DuplicateMatch(
                                innovation1=innovation,
                                innovation2=candidate,
                                similarity_score=float(adjusted_similarity),
                                shared_non_vtt_orgs=list(org_overlap),
                                confidence_level=confidence,
                                match_reasons=match_reasons,
                                url_match=url_match,
                                domain_match=domain_match,
                            )
                            duplicates.append(duplicate)

            progress.advance(task)

    return duplicates


def display_header():
    header_text = Text(
        "🚀 VTT Innovation Analysis with Azure OpenAI 🚀",
        style="bold blue",
        justify="center",
    )
    console.print(Panel(header_text, box=box.DOUBLE, style="blue"))


def display_duplicates(duplicates: List[DuplicateMatch], limit: int = 15):
    if not duplicates:
        console.print(
            Panel("[yellow]No potential duplicates found![/yellow]", title="🎉 Results")
        )
        return

    table = Table(
        title=f"⚠️ Top {limit} Potential Duplicates",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold red",
    )
    table.add_column("Innovation 1", style="green", width=25, overflow="fold")
    table.add_column("Innovation 2", style="green", width=25, overflow="fold")
    table.add_column("Confidence", style="yellow", width=8)
    table.add_column("Similarity", style="cyan", justify="center", width=8)
    table.add_column("Context", style="blue", width=15, overflow="fold")
    table.add_column("Shared Orgs", style="magenta", justify="center", width=8)

    sorted_duplicates = sorted(
        duplicates, key=lambda x: x.similarity_score, reverse=True
    )

    for dup in sorted_duplicates[:limit]:
        confidence_style = {
            "high": "[bold green]HIGH[/bold green]",
            "medium": "[bold yellow]MED[/bold yellow]",
            "low": "[bold red]LOW[/bold red]",
        }.get(dup.confidence_level, dup.confidence_level)

        context_info = []
        if dup.url_match:
            context_info.append("Same URL")
        elif dup.domain_match:
            context_info.append("Same domain")
        if len(dup.shared_non_vtt_orgs) > 1:
            context_info.append(f"{len(dup.shared_non_vtt_orgs)} shared orgs")

        context_str = ", ".join(context_info) if context_info else "Semantic only"

        table.add_row(
            dup.innovation1.name[:22] + "..."
            if len(dup.innovation1.name) > 25
            else dup.innovation1.name,
            dup.innovation2.name[:22] + "..."
            if len(dup.innovation2.name) > 25
            else dup.innovation2.name,
            confidence_style,
            f"{dup.similarity_score:.3f}",
            context_str,
            str(len(dup.shared_non_vtt_orgs)),
        )
    console.print(table)


def export_results(results: AnalysisResults, output_dir: str = "output"):
    """Export analysis results to JSON files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"vtt_azure_analysis_{timestamp}.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results.model_dump(), f, indent=2, default=str, ensure_ascii=False)

    console.print(f"[green]✅ Results exported to {results_file}[/green]")


@app.command()
def analyze(
    vtt_data_dir: str = typer.Option(
        "data/graph_docs_vtt_domain_names_resolved/", help="VTT domain data directory"
    ),
    partner_data_dir: str = typer.Option(
        "data/graph_docs_names_resolved/", help="Partner domain data directory"
    ),
    comp_csv_path: str = typer.Option(
        "data/dataframes/vtt_mentions_comp_domain.csv", help="Company domain CSV file"
    ),
    vtt_csv_path: str = typer.Option(
        "data/dataframes/comp_mentions_vtt_domain.csv", help="VTT domain CSV file"
    ),
    azure_config_path: str = typer.Option(
        "data/keys/azure_config.json", help="Azure OpenAI config file path"
    ),
    model_key: str = typer.Option(
        "gpt-4o-mini", help="Azure OpenAI model key from config"
    ),
    similarity_threshold: float = typer.Option(
        0.80, help="Cosine similarity threshold"
    ),
    min_shared_orgs: int = typer.Option(
        1, help="Min shared non-VTT orgs for duplicates"
    ),
    top_k_candidates: int = typer.Option(100, help="Top K candidates from FAISS"),
    batch_size: int = typer.Option(50, help="Batch size for embeddings"),
    dimensions: int = typer.Option(
        1024, help="Embedding dimensions (1024 for text-embedding-3-large)"
    ),
    cache_embeddings: bool = typer.Option(True, help="Enable embedding caching"),
    cache_dir: str = typer.Option("cache/embeddings", help="Cache directory"),
    export_results_flag: bool = typer.Option(True, help="Export results to JSON"),
    output_dir: str = typer.Option("output", help="Output directory"),
    limit_duplicates: int = typer.Option(15, help="Number of duplicates to display"),
):
    """🚀 Analysis with Azure OpenAI and full context from original data"""

    if not FAISS_AVAILABLE:
        console.print("[red]❌ FAISS is required but not installed.[/red]")
        console.print("[yellow]Install with: pip install faiss-cpu[/yellow]")
        raise typer.Exit(1)

    start_time = time.time()
    display_header()

    # Load original context data
    console.print("[cyan]📊 Loading original context data...[/cyan]")
    comp_context, vtt_context = load_original_context_data(comp_csv_path, vtt_csv_path)

    # Validate data directories
    data_paths = []
    for dir_path, name in [(vtt_data_dir, "VTT"), (partner_data_dir, "Partner")]:
        path = Path(dir_path)
        if path.exists() and path.is_dir() and list(path.glob("*.pkl")):
            data_paths.append(path)
        else:
            console.print(
                f"[yellow]Warning: {name} directory {dir_path} not found or empty. Skipping.[/yellow]"
            )

    if not data_paths:
        console.print("[red]Error: No valid data directories found. Exiting.[/red]")
        raise typer.Exit(1)

    # Setup components
    azure_client = setup_azure_openai_client(azure_config_path, model_key)
    cache_manager = EmbeddingCacheManager(cache_dir) if cache_embeddings else None
    faiss_engine = FAISSSearchEngine()

    if cache_manager:
        console.print(f"[cyan]📦 Embedding cache enabled at {cache_dir}[/cyan]")

    # Load and process data
    console.print("[cyan]📂 Loading graph documents...[/cyan]")
    graph_docs, source_counts = load_and_validate_data(data_paths)

    if not graph_docs:
        console.print("[red]No documents loaded successfully! Exiting.[/red]")
        raise typer.Exit(1)

    console.print(f"[green]✅ Loaded {len(graph_docs)} documents.[/green]")

    # Extract innovations with context
    console.print("[cyan]🧪 Extracting innovations with full context...[/cyan]")
    innovations = extract_innovations_with_context(
        graph_docs, comp_context, vtt_context
    )

    vtt_involved = sum(1 for inn in innovations if inn.has_vtt_involvement)
    console.print(
        f"[green]✅ Extracted {len(innovations)} innovation mentions ({vtt_involved} with VTT involvement).[/green]"
    )

    # Generate embeddings
    console.print(f"[cyan]🧠 Generating embeddings with Azure OpenAI...[/cyan]")
    embeddings_map = generate_embeddings_batch(
        innovations, azure_client, batch_size, dimensions, cache_manager
    )
    console.print(f"[green]✅ Generated {len(embeddings_map)} embeddings.[/green]")

    # Build FAISS index
    console.print("[cyan]⚡ Building FAISS index...[/cyan]")
    faiss_build_time = faiss_engine.build_index(innovations, embeddings_map)

    # Find duplicates with context
    console.print(
        f"[cyan]🚀 Finding duplicates with context (threshold: {similarity_threshold})...[/cyan]"
    )
    duplicates = find_duplicates_with_context(
        innovations,
        embeddings_map,
        faiss_engine,
        similarity_threshold,
        min_shared_orgs,
        top_k_candidates,
    )

    processing_time = time.time() - start_time

    # Create results
    config = {
        "similarity_threshold": similarity_threshold,
        "min_shared_orgs": min_shared_orgs,
        "embedding_model": azure_client.embedding_deployment,
        "embedding_dimensions": dimensions,
        "top_k_candidates": top_k_candidates,
        "batch_size": batch_size,
        "cache_embeddings": cache_embeddings,
        "uses_original_context": True,
        "filters_vtt": False,
        "provider": "azure_openai",
    }

    results = AnalysisResults(
        total_raw_mentions=len(innovations),
        unique_innovations=len(set(inn.name.lower() for inn in innovations)),
        duplicates=duplicates,
        source_counts=source_counts,
        processing_time=processing_time,
        config=config,
    )

    # Display results
    console.print()
    display_duplicates(duplicates, limit_duplicates)

    # Summary
    console.print(f"\n[green]🚀 Analysis Complete![/green]")
    console.print(f"[cyan]• Total duplicates found: {len(duplicates)}[/cyan]")
    console.print(
        f"[cyan]• High confidence: {sum(1 for d in duplicates if d.confidence_level == 'high')}[/cyan]"
    )
    console.print(
        f"[cyan]• URL matches: {sum(1 for d in duplicates if d.url_match)}[/cyan]"
    )
    console.print(
        f"[cyan]• Domain matches: {sum(1 for d in duplicates if d.domain_match)}[/cyan]"
    )
    console.print(f"[cyan]• Processing time: {processing_time:.2f}s[/cyan]")
    console.print(f"[cyan]• Index build time: {faiss_build_time:.2f}s[/cyan]")

    # Export results
    if export_results_flag:
        console.print(f"\n[cyan]📤 Exporting results...[/cyan]")
        export_results(results, output_dir)


@app.command()
def info():
    """ℹ️ Display tool information"""
    info_text = """
    🚀 [yellow]VTT Innovation Analysis with Azure OpenAI[/yellow]
    
    ⚡ [yellow]Key Features:[/yellow]
    • Uses Azure OpenAI for high-quality embeddings
    • Uses ALL original CSV metadata (URLs, titles, dates, types)
    • Keeps VTT in analysis (no filtering out)
    • Context-based duplicate detection
    • URL and domain matching for high confidence
    • Shared non-VTT organization analysis
    
    🔍 [yellow]How it works:[/yellow]
    • Loads original CSV context data
    • Generates embeddings using innovation names + descriptions + titles
    • Uses FAISS for fast similarity search
    • Boosts confidence with URL/domain/org context
    • Tracks VTT involvement without filtering
    
    [cyan]Requirements:[/cyan] pip install faiss-cpu openai pandas
    [cyan]Config:[/cyan] Requires Azure OpenAI config file with API keys
    """
    console.print(
        Panel(info_text, title="🚀 VTT Analysis with Azure OpenAI", style="blue")
    )


if __name__ == "__main__":
    app()
