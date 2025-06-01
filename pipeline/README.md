# VTT Innovation Disambiguation Pipeline

A complete pipeline for resolving ambiguous innovation entities in VTT's knowledge graph using Azure OpenAI embeddings, FAISS similarity search, and a Model Context Protocol (MCP) server for graph curation.

## The Challenge

VTT's innovation data exists as a fragmented knowledge graph where the same real-world innovation appears as multiple distinct nodes:

- "Solar Foods protein production technology"
- "Neo-Carbon Food protein production process"
- "Protein from air by VTT & LUT"

This fragmentation obscures true collaboration patterns and innovation portfolios.

## The Solution: Three-Stage Pipeline

### Stage 1: Ambiguity Analysis (`scripts/innovations_analysis.py`)

**Purpose:** Identify potential duplicate innovation mentions using semantic similarity

**Features:**

- Processes graph documents from multiple data sources
- Generates Azure OpenAI embeddings with full context (names, descriptions, URLs, metadata)
- Uses FAISS for high-performance similarity search
- Context-aware duplicate detection with URL/domain matching
- Embedding caching for efficiency

```bash
# Run full analysis
uv run python scripts/innovations_analysis.py analyze \
    --vtt-data-dir "data/graph_docs_vtt_domain/" \
    --partner-data-dir "data/graph_docs_partners/" \
    --similarity-threshold 0.80 \
    --batch-size 50

# View tool information
uv run python scripts/innovations_analysis.py info
```

### Stage 2: LLM Agent + MCP Server (`agents/innovation-curator-agent.ts`)

**Purpose:** Intelligent disambiguation using LLM reasoning with MCP graph operations

**LLM Agent:**

- Analyzes candidate pairs using GPT-4.1 with rich context
- Makes SAME/DIFFERENT decisions with detailed reasoning
- Orchestrates MCP tools for graph updates
- Tracks decisions in JSONL format for audit trails

**MCP Server Tools:**

- `resolve_or_create_canonical_innovation` - Smart innovation entity resolution
- `resolve_or_create_organization` - Organization management by VAT ID
- `add_mention_to_link` - Link organizations to innovations with full provenance
- `search_similar_innovations` - Semantic search using embeddings
- `merge_innovations` - Consolidate duplicate innovations

```bash
# Start MCP server
uv run fastmcp run mcp/innovation_entity_server.py --transport sse --port 9000

# Run LLM curation agent
bun run agents/innovation-curator-agent.ts duplicates.json
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

# Install Bun for the LLM agent
curl -fsSL https://bun.sh/install | bash

# Sync Python dependencies
uv sync

# Install TypeScript agent dependencies
bun install
```

### 2. Setup Environment

```bash
# Memgraph Configuration
export MEMGRAPH_HOST="bolt://localhost:7687"

# Azure OpenAI Configuration
export AZURE_CONFIG_PATH="/path/to/azure_config.json"
export AZURE_MODEL_KEY="gpt-4.1-mini"
export EMBEDDING_DIMENSION="1024"

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

## Key Dependencies

- `fastmcp` - MCP server framework
- `openai` - Azure OpenAI client
- `faiss-cpu` - Similarity search
- `neo4j` - Memgraph driver
- `pandas` - Data processing
- `typer` - CLI framework
- `rich` - Terminal UI

## Data Requirements

- Graph documents in `.pkl` format (from GraphRAG/LangChain)
- CSV files with innovation metadata (URLs, titles, dates)
- JSON organization glossary for entity resolution
