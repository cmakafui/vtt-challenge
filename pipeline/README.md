# VTT Innovation Disambiguation Pipeline

A complete pipeline for resolving ambiguous innovation entities in VTT's knowledge graph using Azure OpenAI embeddings, FAISS similarity search, and a Model Context Protocol (MCP) server for graph curation.

## The Challenge

VTT's innovation data exists as a fragmented knowledge graph where the same real-world innovation appears as multiple distinct nodes:

- "Solar Foods protein production technology"
- "Neo-Carbon Food protein production process"
- "Protein from air by VTT & LUT"

This fragmentation obscures true collaboration patterns and innovation portfolios.

## The Solution: Three-Stage Pipeline

### Stage 1: Ambiguity Analysis (`scripts/ambiguity_analysis.py`)

**Purpose:** Identify potential duplicate innovation mentions using semantic similarity

**Features:**

- Processes graph documents from multiple data sources
- Generates Azure OpenAI embeddings with full context (names, descriptions, URLs, metadata)
- Uses FAISS for high-performance similarity search
- Context-aware duplicate detection with URL/domain matching
- Embedding caching for efficiency

```bash
# Run full analysis
uv run python scripts/ambiguity_analysis.py analyze \
    --vtt-data-dir "data/graph_docs_vtt_domain/" \
    --partner-data-dir "data/graph_docs_partners/" \
    --similarity-threshold 0.80 \
    --batch-size 50

# View tool information
uv run python scripts/ambiguity_analysis.py info
```

### Stage 2: MCP Server (`mcp/innovation_entity_server.py`)

**Purpose:** Intelligent graph curation and canonical entity management

**Core Tools:**

- `resolve_or_create_canonical_innovation` - Smart innovation entity resolution
- `resolve_or_create_organization` - Organization management by VAT ID
- `add_mention_to_link` - Link organizations to innovations with full provenance
- `search_similar_innovations` - Semantic search using embeddings
- `merge_innovations` - Consolidate duplicate innovations
- `get_innovation_timeline` - Track innovation mention history

```bash
# Start MCP server
uv run fastmcp run mcp/innovation_entity_server.py --transport sse --port 9000
```

### Stage 3: Entity Ingestion (`scripts/ingest_entities.py`)

**Purpose:** High-performance batch loading of resolved entities into Memgraph

**Features:**

- Optimized batch operations for maximum throughput
- Handles organization glossaries and duplicate resolution results
- Rich progress reporting and error handling

```bash
# Batch ingest organizations
uv run python scripts/ingest_entities.py ingest \
    glossary.json \
    duplicates.json \
    --batch-size 100
```

## Quick Start

### 1. Install Dependencies

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### 2. Setup Environment

```bash
# Memgraph Configuration
export MEMGRAPH_HOST="bolt://localhost:7687"

# Azure OpenAI Configuration
export AZURE_CONFIG_PATH="/path/to/azure_config.json"
export AZURE_MODEL_KEY="gpt-4o-mini"
export EMBEDDING_DIMENSION="2048"

# MCP Server Configuration
export FASTMCP_SERVER_PORT="9000"
```

### 3. Start Services

```bash
# Start Memgraph database
docker compose up -d

# Start MCP server
uv run fastmcp run mcp/innovation_entity_server.py --transport sse --port 9000
```

## Azure OpenAI Configuration

Create `azure_config.json`:

```json
{
  "gpt-4o-mini": {
    "api_key": "your-azure-openai-api-key",
    "api_version": "2024-02-01",
    "api_base": "https://your-instance.openai.azure.com/",
    "emb_deployment": "your-embedding-deployment-name"
  }
}
```

## Pipeline Output

The pipeline transforms fragmented innovation mentions into a clean canonical knowledge graph:

**Before:** Multiple nodes for same innovation

- "Solar Foods protein tech" (Node A)
- "Neo-Carbon Food process" (Node B)
- "Air protein technology" (Node C)

**After:** Single canonical innovation node

- "Advanced Protein Synthesis from CO2" (UUID: abc-123)
  - With rich provenance tracking all original mentions
  - Semantic embeddings for similarity search
  - Full audit trail to source URLs and publication dates

## Services

- **Memgraph**: Graph database at `localhost:7687`
- **Memgraph Lab**: Web UI at `http://localhost:3000`
- **MCP Server**: Entity management at `localhost:9000`

## Data Requirements

- Graph documents in `.pkl` format (from GraphRAG/LangChain)
- CSV files with innovation metadata (URLs, titles, dates)
- JSON organization glossary for entity resolution
