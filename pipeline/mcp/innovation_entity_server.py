#!/usr/bin/env python
# /// script
# dependencies = [
#   "neo4j",
#   "openai",
#   "fastmcp",
#   "pydantic",
# ]
# ///

# innovation_entity_server.py
import sys
import traceback

try:
    import logging

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logger = logging.getLogger("innovation_entity_server")
    logger.info("Starting server initialization...")

    from fastmcp import FastMCP, Context
    from neo4j import GraphDatabase
    from contextlib import asynccontextmanager
    from dataclasses import dataclass
    from typing import List, Dict, Any, Optional
    import os
    import json
    from collections.abc import AsyncIterator
    import hashlib
    from openai import AzureOpenAI
    from models.schemas import (
        MentionRecord,
        InnovationResponse,
        OrganizationResponse,
        InvolvementResponse,
        SimilarityResponse,
        MergeInnovationResponse,
        MergeOrganizationResponse,
        TimelineMention,
        TimelineResponse,
    )

except Exception as e:
    print(f"ERROR DURING INITIALIZATION: {str(e)}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# AZURE OPENAI EMBEDDINGS CLIENT
# ═══════════════════════════════════════════════════════════════════════════════


class AzureOpenAIEmbeddings:
    def __init__(
        self,
        config_path: str = "data/keys/azure_config.json",
        model_key: str = "gpt-4.1-mini",
    ):
        """Initialize Azure OpenAI client for embeddings."""
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.model_config = self.config[model_key]

        self.client = AzureOpenAI(
            api_key=self.model_config["api_key"],
            api_version=self.model_config["api_version"],
            azure_endpoint=self.model_config["api_base"],
        )

        self.embedding_deployment = self.model_config["emb_deployment"]

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text string."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_deployment,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return [0.0] * 2048  # Return zero vector fallback


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MemgraphContext:
    """Context for Memgraph connection with Azure OpenAI embeddings."""

    driver: GraphDatabase.driver
    embeddings_client: AzureOpenAIEmbeddings
    embedding_dimension: int = 2048  # Azure OpenAI text-embedding-ada-002 dimension
    vector_index_name: str = "innovation_embeddings"
    innovation_label: str = "Innovation"
    organization_label: str = "Organization"


# ═══════════════════════════════════════════════════════════════════════════════
# LIFESPAN MANAGER
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def memgraph_lifespan(server: FastMCP) -> AsyncIterator[MemgraphContext]:
    """Initialize and manage Memgraph connection with Azure OpenAI."""
    # Configuration
    host = os.getenv("MEMGRAPH_HOST", "bolt://localhost:7687")
    user = os.getenv("MEMGRAPH_USER", "")
    password = os.getenv("MEMGRAPH_PASSWORD", "")
    azure_config_path = os.getenv("AZURE_CONFIG_PATH")
    model_key = os.getenv("AZURE_MODEL_KEY", "gpt-4o-mini")
    embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "2048"))

    # Initialize connections
    driver = GraphDatabase.driver(host, auth=(user, password))
    embeddings_client = AzureOpenAIEmbeddings(azure_config_path, model_key)

    try:
        # Verify Memgraph connection
        driver.verify_connectivity()
        logger.info(f"Connected to Memgraph at {host}")

        # Test Azure OpenAI connection
        test_embedding = embeddings_client.get_embedding("Test embedding generation")
        embedding_dimension = len(test_embedding)
        logger.info(
            f"Connected to Azure OpenAI, embedding dimension: {embedding_dimension}"
        )

        # Create vector index if it doesn't exist
        with driver.session() as session:
            result = session.run(
                """
                CALL vector_search.show_index_info() YIELD *
                WITH * WHERE index_name = $index_name
                RETURN count(*) as count
                """,
                {"index_name": "innovation_embeddings"},
            )

            index_exists = result.single()["count"] > 0

            if not index_exists:
                logger.info("Creating vector index for innovation embeddings...")
                session.run(f"""
                    CREATE VECTOR INDEX innovation_embeddings 
                    ON :Innovation(embedding) 
                    WITH CONFIG {{
                        "dimension": {embedding_dimension},
                        "capacity": 10000,
                        "metric": "cos"
                    }};
                """)
                logger.info("Vector index created successfully")

        # Yield context to server
        yield MemgraphContext(
            driver=driver,
            embeddings_client=embeddings_client,
            embedding_dimension=embedding_dimension,
        )

    finally:
        # Clean up connections
        if driver is not None:
            try:
                await driver.close() if hasattr(driver, "close") else driver.close()
                logger.info("Memgraph connection closed")
            except Exception as e:
                logger.error(f"Error closing Memgraph connection: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def generate_embedding(ctx: Context, text: str) -> List[float]:
    """Generate embedding for given text using Azure OpenAI."""
    memgraph = ctx.request_context.lifespan_context
    try:
        return memgraph.embeddings_client.get_embedding(text)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return [0.0] * memgraph.embedding_dimension


# ═══════════════════════════════════════════════════════════════════════════════
# CREATE MCP SERVER
# ═══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP("VTTInnovationEntityServer", lifespan=memgraph_lifespan)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE INNOVATION ENTITY MANAGEMENT TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
def resolve_or_create_canonical_innovation(
    ctx: Context,
    name: str,
    description: str,
    mention_context: Dict[str, Any],
    similarity_threshold: float = 0.85,
) -> InnovationResponse:
    """Resolve or create a canonical innovation entity with advanced matching capabilities.

    This tool performs intelligent entity resolution for innovations using a combination of:
    - Exact name matching
    - Vector similarity search using Azure OpenAI embeddings
    - Fuzzy matching for name variations

    If no high-confidence match is found, it creates a new canonical innovation with:
    - A unique innovation ID
    - The provided name and description
    - Vector embeddings for future similarity matching
    - Timestamps for creation and updates

    Args:
        name: The innovation name to resolve or create
        description: Detailed description of the innovation
        mention_context: Additional context about the mention (organizations, source, etc.)
        similarity_threshold: Minimum similarity score (0-1) required for considering a match

    Returns:
        InnovationResponse containing:
        - Success status and any errors
        - Innovation ID and canonical name
        - Aggregated description
        - Creation status
        - List of similar matches with confidence scores
    """
    memgraph = ctx.request_context.lifespan_context

    # Generate embedding for search
    search_text = f"{name} {description}"
    embedding = generate_embedding(ctx, search_text)

    with memgraph.driver.session() as session:
        # First try exact name matching
        exact_matches = session.run(
            """
            MATCH (i:Innovation)
            WHERE i.canonical_name = $name OR $name IN i.name_variations
            RETURN i.innovation_id as innovation_id, i.canonical_name as canonical_name,
                   i.aggregated_description as description, 1.0 as confidence
            LIMIT 3
            """,
            {"name": name},
        )

        exact_results = list(exact_matches)

        # Use vector search for semantic matching
        vector_matches = session.run(
            """
            CALL vector_search.search($index_name, 10, $embedding)
            YIELD node, similarity
            WITH node, similarity
            WHERE similarity >= $threshold
            RETURN node.innovation_id as innovation_id, 
                   node.canonical_name as canonical_name,
                   node.aggregated_description as description,
                   similarity as confidence
            ORDER BY similarity DESC
            """,
            {
                "index_name": memgraph.vector_index_name,
                "embedding": embedding,
                "threshold": similarity_threshold,
            },
        )

        vector_results = list(vector_matches)

        # Combine results, prioritizing exact matches
        all_results = exact_results + vector_results
        all_results.sort(key=lambda r: r["confidence"], reverse=True)

        # If high-confidence match found, return it
        if all_results and all_results[0]["confidence"] >= similarity_threshold:
            top_match = all_results[0]
            return InnovationResponse(
                success=True,
                innovation_id=top_match["innovation_id"],
                canonical_name=top_match["canonical_name"],
                aggregated_description=top_match["description"],
                created=False,
                similarity_matches=[
                    {
                        "innovation_id": r["innovation_id"],
                        "canonical_name": r["canonical_name"],
                        "confidence": r["confidence"],
                    }
                    for r in all_results[:5]
                ],
            )

        # No good match found, create new canonical innovation
        innovation_id = f"innov_{hashlib.md5(name.lower().encode()).hexdigest()[:12]}"

        # Store name variations for future matching
        name_variations = [name]
        if name != name.lower():
            name_variations.append(name.lower())

        result = session.run(
            """
            CREATE (i:Innovation {
                innovation_id: $innovation_id,
                canonical_name: $canonical_name,
                aggregated_description: $description,
                name_variations: $name_variations,
                embedding: $embedding,
                created_at: datetime(),
                updated_at: datetime()
            })
            RETURN i.innovation_id as innovation_id, i.canonical_name as canonical_name,
                   i.aggregated_description as description
            """,
            {
                "innovation_id": innovation_id,
                "canonical_name": name,
                "description": description,
                "name_variations": name_variations,
                "embedding": embedding,
            },
        )

        record = result.single()

        return InnovationResponse(
            success=True,
            innovation_id=record["innovation_id"],
            canonical_name=record["canonical_name"],
            aggregated_description=record["description"],
            created=True,
            similarity_matches=[],
        )


@mcp.tool()
def resolve_or_create_organization(
    ctx: Context, vat_id: str, name_hint: Optional[str] = None
) -> OrganizationResponse:
    """Resolve or create an organization entity using VAT ID as the primary identifier.

    This tool manages organization entities in the knowledge graph, ensuring:
    - Unique identification through VAT ID
    - Proper name canonicalization
    - Alias management for alternative names
    - Timestamp tracking for entity lifecycle

    Args:
        vat_id: The VAT ID of the organization (primary identifier)
        name_hint: Optional name to use if creating a new organization

    Returns:
        OrganizationResponse containing:
        - Success status and any errors
        - VAT ID and canonical name
        - List of known aliases
        - Whether the organization was found or newly created
    """
    memgraph = ctx.request_context.lifespan_context

    with memgraph.driver.session() as session:
        # Look for existing organization
        result = session.run(
            """
            MATCH (o:Organization {vat_id: $vat_id})
            RETURN o.vat_id as vat_id, o.canonical_name as canonical_name,
                   o.aliases as aliases
            """,
            {"vat_id": vat_id},
        )

        record = result.single()

        if record:
            return OrganizationResponse(
                success=True,
                vat_id=record["vat_id"],
                canonical_name=record["canonical_name"],
                aliases=record["aliases"] or [],
                found=True,
            )

        # Organization not found - create minimal entry if name_hint provided
        if name_hint:
            session.run(
                """
                CREATE (o:Organization {
                    vat_id: $vat_id,
                    canonical_name: $name_hint,
                    aliases: [],
                    created_at: datetime(),
                    updated_at: datetime()
                })
                """,
                {"vat_id": vat_id, "name_hint": name_hint},
            )

            return OrganizationResponse(
                success=True,
                vat_id=vat_id,
                canonical_name=name_hint,
                aliases=[],
                found=False,
            )

        return OrganizationResponse(
            success=False,
            error=f"Organization with VAT ID {vat_id} not found and no name hint provided",
        )


@mcp.tool()
def add_mention_to_link(
    ctx: Context, org_vat_id: str, innovation_id: str, mention_record: MentionRecord
) -> InvolvementResponse:
    """Add a mention record to an organization-innovation relationship.

    This tool manages the relationship between organizations and innovations by:
    - Creating or updating INVOLVED_IN relationships
    - Maintaining a list of mentions with full provenance
    - Tracking primary roles and relationship descriptions
    - Managing relationship fingerprints for deduplication
    - Updating timestamps for relationship lifecycle

    Args:
        org_vat_id: VAT ID of the organization
        innovation_id: ID of the innovation
        mention_record: Detailed record of the mention including:
            - Unique mention ID
            - Original name and text
            - Source information (URL, doc ID, dataset)
            - Publication date
            - Role and relationship description
            - Confidence score

    Returns:
        InvolvementResponse containing:
        - Success status and any errors
        - Relationship fingerprint
        - Organization and innovation identifiers
        - Primary role
        - Status (created/updated/duplicate)
        - Total mentions count
    """
    memgraph = ctx.request_context.lifespan_context

    # Convert Pydantic model to dict
    mention_dict = mention_record.model_dump()

    with memgraph.driver.session() as session:
        # Verify both entities exist
        verification = session.run(
            """
            MATCH (o:Organization {vat_id: $vat_id})
            MATCH (i:Innovation {innovation_id: $innovation_id})
            RETURN o.canonical_name as org_name, i.canonical_name as innovation_name
            """,
            {"vat_id": org_vat_id, "innovation_id": innovation_id},
        )

        verification_record = verification.single()
        if not verification_record:
            return InvolvementResponse(
                success=False, error="Organization or Innovation not found"
            )

        org_name = verification_record["org_name"]
        innovation_name = verification_record["innovation_name"]

        # Generate relationship fingerprint
        fingerprint = f"involved_{org_vat_id}_{innovation_id}"

        # Check if relationship exists
        rel_check = session.run(
            """
            MATCH (o:Organization {vat_id: $vat_id})-[r:INVOLVED_IN]->(i:Innovation {innovation_id: $innovation_id})
            RETURN r.mentions as mentions, r.primary_role as primary_role
            """,
            {"vat_id": org_vat_id, "innovation_id": innovation_id},
        )

        rel_record = rel_check.single()

        if rel_record:
            # Add mention to existing relationship
            result = session.run(
                """
                MATCH (o:Organization {vat_id: $vat_id})-[r:INVOLVED_IN]->(i:Innovation {innovation_id: $innovation_id})
                SET r.mentions = r.mentions + [$mention],
                    r.updated_at = datetime(),
                    r.primary_role = COALESCE(r.primary_role, $role)
                RETURN size(r.mentions) as mentions_count, r.primary_role as primary_role
                """,
                {
                    "vat_id": org_vat_id,
                    "innovation_id": innovation_id,
                    "mention": mention_dict,
                    "role": mention_record.role_in_mention or "Collaborator",
                },
            )

            record = result.single()

            return InvolvementResponse(
                success=True,
                fingerprint=fingerprint,
                org_vat_id=org_vat_id,
                innovation_id=innovation_id,
                org_name=org_name,
                innovation_name=innovation_name,
                primary_role=record["primary_role"],
                status="updated",
                mentions_count=record["mentions_count"],
            )

        else:
            # Create new relationship
            result = session.run(
                """
                MATCH (o:Organization {vat_id: $vat_id})
                MATCH (i:Innovation {innovation_id: $innovation_id})
                CREATE (o)-[r:INVOLVED_IN {
                    fingerprint: $fingerprint,
                    primary_role: $role,
                    mentions: [$mention],
                    created_at: datetime(),
                    updated_at: datetime()
                }]->(i)
                RETURN size(r.mentions) as mentions_count, r.primary_role as primary_role
                """,
                {
                    "vat_id": org_vat_id,
                    "innovation_id": innovation_id,
                    "fingerprint": fingerprint,
                    "role": mention_record.role_in_mention or "Collaborator",
                    "mention": mention_dict,
                },
            )

            record = result.single()

            return InvolvementResponse(
                success=True,
                fingerprint=fingerprint,
                org_vat_id=org_vat_id,
                innovation_id=innovation_id,
                org_name=org_name,
                innovation_name=innovation_name,
                primary_role=record["primary_role"],
                status="created",
                mentions_count=record["mentions_count"],
            )


@mcp.tool()
def search_similar_innovations(
    ctx: Context, query_text: str, threshold: float = 0.85, limit: int = 10
) -> SimilarityResponse:
    """Search for semantically similar innovations using vector embeddings.

    This tool performs semantic search using Azure OpenAI embeddings to find:
    - Innovations with similar meaning or context
    - Results ranked by similarity score
    - Configurable threshold and result limit

    Args:
        query_text: The text to search for similar innovations
        threshold: Minimum similarity score (0-1) for results
        limit: Maximum number of results to return

    Returns:
        SimilarityResponse containing:
        - Original query text
        - List of matching innovations with:
            - Innovation ID
            - Canonical name
            - Description
            - Confidence score
        - Total count of matches
    """
    memgraph = ctx.request_context.lifespan_context

    # Generate embedding for query
    embedding = generate_embedding(ctx, query_text)

    with memgraph.driver.session() as session:
        results = session.run(
            """
            CALL vector_search.search($index_name, $limit, $embedding)
            YIELD node, similarity
            WITH node, similarity
            WHERE similarity >= $threshold
            RETURN node.innovation_id as innovation_id,
                   node.canonical_name as canonical_name,
                   node.aggregated_description as description,
                   similarity as confidence
            ORDER BY similarity DESC
            """,
            {
                "index_name": memgraph.vector_index_name,
                "limit": limit,
                "embedding": embedding,
                "threshold": threshold,
            },
        )

        innovations = []
        for record in results:
            innovations.append(
                {
                    "innovation_id": record["innovation_id"],
                    "canonical_name": record["canonical_name"],
                    "description": record["description"],
                    "confidence": record["confidence"],
                }
            )

        return SimilarityResponse(
            query=query_text, results=innovations, count=len(innovations)
        )


@mcp.tool()
def get_random_organization(ctx: Context, limit: int = 1) -> OrganizationResponse:
    """Retrieve random organization(s) from the knowledge graph.

    This tool provides a way to sample organizations from the graph, useful for:
    - Testing and development
    - Data exploration
    - Random sampling for analysis

    Args:
        limit: Number of random organizations to retrieve

    Returns:
        OrganizationResponse containing:
        - Success status and any errors
        - Organization details (VAT ID, name, aliases)
        - Whether organizations were found
    """
    memgraph = ctx.request_context.lifespan_context

    with memgraph.driver.session() as session:
        result = session.run(
            """
            MATCH (o:Organization)
            WITH o, rand() AS r
            ORDER BY r
            RETURN o.vat_id as vat_id, 
                   o.canonical_name as canonical_name,
                   o.aliases as aliases
            LIMIT $limit
            """,
            {"limit": limit},
        )

        record = result.single()

        if record:
            return OrganizationResponse(
                success=True,
                vat_id=record["vat_id"],
                canonical_name=record["canonical_name"],
                aliases=record["aliases"] or [],
                found=True,
            )

        return OrganizationResponse(
            success=False, error="No organizations found in database"
        )


@mcp.tool()
def find_duplicate_innovations(ctx: Context) -> Dict[str, Any]:
    """Find innovation nodes with duplicate names."""
    memgraph = ctx.request_context.lifespan_context

    with memgraph.driver.session() as session:
        result = session.run("""
            MATCH (i:Innovation)
            WHERE i.canonical_name IS NOT NULL
            WITH i.canonical_name AS name, collect(i) AS innovations, count(*) AS count
            WHERE count > 1
            RETURN name, count, innovations
        """)

        duplicates = []
        for record in result:
            name = record["name"]
            count = record["count"]
            innovations = record["innovations"]

            innovation_details = [
                {
                    "innovation_id": innovation["innovation_id"],
                    "canonical_name": innovation["canonical_name"],
                    "aggregated_description": innovation.get(
                        "aggregated_description", ""
                    )[:100]
                    + "...",
                }
                for innovation in innovations
            ]

            duplicates.append(
                {"name": name, "count": count, "innovations": innovation_details}
            )

        return {"total_duplicate_groups": len(duplicates), "duplicates": duplicates}


@mcp.tool()
def find_duplicate_organizations(ctx: Context) -> Dict[str, Any]:
    """Find organization nodes with duplicate names."""
    memgraph = ctx.request_context.lifespan_context

    with memgraph.driver.session() as session:
        result = session.run("""
            MATCH (o:Organization)
            WHERE o.canonical_name IS NOT NULL
            WITH o.canonical_name AS name, collect(o) AS organizations, count(*) AS count
            WHERE count > 1
            RETURN name, count, organizations
        """)

        duplicates = []
        for record in result:
            name = record["name"]
            count = record["count"]
            organizations = record["organizations"]

            org_details = [
                {
                    "vat_id": org["vat_id"],
                    "canonical_name": org["canonical_name"],
                    "aliases": org.get("aliases", []),
                }
                for org in organizations
            ]

            duplicates.append(
                {"name": name, "count": count, "organizations": org_details}
            )

        return {"total_duplicate_groups": len(duplicates), "duplicates": duplicates}


@mcp.tool()
def merge_innovations(
    ctx: Context, source_innovation_id: str, target_innovation_id: str
) -> MergeInnovationResponse:
    """Merge source innovation into target innovation. Source will be deleted."""
    memgraph = ctx.request_context.lifespan_context

    with memgraph.driver.session() as session:
        # Verify both exist
        verification = session.run(
            """
            MATCH (source:Innovation {innovation_id: $source_id})
            MATCH (target:Innovation {innovation_id: $target_id})
            RETURN source.canonical_name as source_name, target.canonical_name as target_name
        """,
            {"source_id": source_innovation_id, "target_id": target_innovation_id},
        )

        verification_record = verification.single()
        if not verification_record:
            return MergeInnovationResponse(
                success=False, error="One or both innovations not found"
            )

        source_name = verification_record["source_name"]
        target_name = verification_record["target_name"]

        # Merge name variations
        session.run(
            """
            MATCH (source:Innovation {innovation_id: $source_id})
            MATCH (target:Innovation {innovation_id: $target_id})
            SET target.name_variations = target.name_variations + source.name_variations,
                target.updated_at = datetime()
        """,
            {"source_id": source_innovation_id, "target_id": target_innovation_id},
        )

        # Transfer relationships from organizations
        session.run(
            """
            MATCH (source:Innovation {innovation_id: $source_id})
            MATCH (target:Innovation {innovation_id: $target_id})
            MATCH (org:Organization)-[r:INVOLVED_IN]->(source)
            
            // Create new relationship to target
            CREATE (org)-[newRel:INVOLVED_IN {
                fingerprint: replace(r.fingerprint, source.innovation_id, target.innovation_id),
                primary_role: r.primary_role,
                mentions: r.mentions,
                created_at: r.created_at,
                updated_at: datetime()
            }]->(target)
        """,
            {"source_id": source_innovation_id, "target_id": target_innovation_id},
        )

        # Delete source and its relationships
        session.run(
            """
            MATCH (source:Innovation {innovation_id: $source_id})
            DETACH DELETE source
        """,
            {"source_id": source_innovation_id},
        )

        return MergeInnovationResponse(
            success=True,
            innovation_id=target_innovation_id,
            canonical_name=target_name,
            merged_from=source_name,
            merged_into=target_name,
        )


@mcp.tool()
def merge_organizations(
    ctx: Context, source_vat_id: str, target_vat_id: str
) -> MergeOrganizationResponse:
    """Merge source organization into target organization. Source will be deleted."""
    memgraph = ctx.request_context.lifespan_context

    with memgraph.driver.session() as session:
        # Verify both exist
        verification = session.run(
            """
            MATCH (source:Organization {vat_id: $source_id})
            MATCH (target:Organization {vat_id: $target_id})
            RETURN source.canonical_name as source_name, target.canonical_name as target_name
        """,
            {"source_id": source_vat_id, "target_id": target_vat_id},
        )

        verification_record = verification.single()
        if not verification_record:
            return MergeOrganizationResponse(
                success=False, error="One or both organizations not found"
            )

        source_name = verification_record["source_name"]
        target_name = verification_record["target_name"]

        # Merge aliases
        session.run(
            """
            MATCH (source:Organization {vat_id: $source_id})
            MATCH (target:Organization {vat_id: $target_id})
            SET target.aliases = target.aliases + source.aliases,
                target.updated_at = datetime()
        """,
            {"source_id": source_vat_id, "target_id": target_vat_id},
        )

        # Transfer relationships to innovations
        session.run(
            """
            MATCH (source:Organization {vat_id: $source_id})
            MATCH (target:Organization {vat_id: $target_id})
            MATCH (source)-[r:INVOLVED_IN]->(innovation:Innovation)
            
            // Create new relationship from target
            CREATE (target)-[newRel:INVOLVED_IN {
                fingerprint: replace(r.fingerprint, source.vat_id, target.vat_id),
                primary_role: r.primary_role,
                mentions: r.mentions,
                created_at: r.created_at,
                updated_at: datetime()
            }]->(innovation)
        """,
            {"source_id": source_vat_id, "target_id": target_vat_id},
        )

        # Delete source and its relationships
        session.run(
            """
            MATCH (source:Organization {vat_id: $source_id})
            DETACH DELETE source
        """,
            {"source_id": source_vat_id},
        )

        return MergeOrganizationResponse(
            success=True,
            vat_id=target_vat_id,
            canonical_name=target_name,
            merged_from=source_name,
            merged_into=target_name,
        )


@mcp.tool()
def get_innovation_timeline(ctx: Context, innovation_id: str) -> TimelineResponse:
    """Get chronological timeline of all mentions for a specific innovation.

    This tool retrieves all mentions of an innovation from relationships with
    organizations, sorted by publication date to show the evolution and
    development of the innovation over time.

    Args:
        innovation_id: The ID of the innovation to get timeline for

    Returns:
        TimelineResponse containing chronologically sorted mentions with
        organization context and date range information.
    """
    memgraph = ctx.request_context.lifespan_context

    with memgraph.driver.session() as session:
        # First, verify the innovation exists and get its details
        innovation_check = session.run(
            """
            MATCH (i:Innovation {innovation_id: $innovation_id})
            RETURN i.innovation_id as innovation_id,
                   i.canonical_name as canonical_name,
                   i.aggregated_description as aggregated_description,
                   i.created_at as created_at,
                   i.updated_at as updated_at
        """,
            {"innovation_id": innovation_id},
        )

        innovation_record = innovation_check.single()
        if not innovation_record:
            return TimelineResponse(
                success=False, error=f"Innovation with ID '{innovation_id}' not found"
            )

        # Get all mentions from relationships
        mentions_query = session.run(
            """
            MATCH (org:Organization)-[r:INVOLVED_IN]->(i:Innovation {innovation_id: $innovation_id})
            RETURN org.vat_id as org_vat_id,
                   org.canonical_name as org_name,
                   r.mentions as mentions
        """,
            {"innovation_id": innovation_id},
        )

        # Collect all mentions with organization context
        all_mentions = []
        organizations_involved = set()

        for record in mentions_query:
            org_vat_id = record["org_vat_id"]
            org_name = record["org_name"]
            mentions = record["mentions"] or []

            organizations_involved.add((org_vat_id, org_name))

            for mention in mentions:
                timeline_mention = TimelineMention(
                    mention_unique_id=mention.get("mention_unique_id", ""),
                    publication_date=mention.get("publication_date"),
                    original_name=mention.get("original_name", ""),
                    source_url=mention.get("source_url"),
                    source_doc_id=mention.get("source_doc_id", ""),
                    dataset_origin=mention.get("dataset_origin", ""),
                    role_in_mention=mention.get("role_in_mention"),
                    relationship_description_in_mention=mention.get(
                        "relationship_description_in_mention"
                    ),
                    original_text=mention.get("original_text"),
                    confidence=mention.get("confidence", 1.0),
                    organization_vat_id=org_vat_id,
                    organization_name=org_name,
                )
                all_mentions.append(timeline_mention)

        # Sort mentions by publication date (handle None dates)
        def sort_key(mention):
            if mention.publication_date:
                try:
                    # Try to parse date for proper sorting
                    from datetime import datetime

                    return datetime.fromisoformat(
                        mention.publication_date.replace("Z", "+00:00")
                    )
                except:
                    # If parsing fails, sort lexicographically
                    return mention.publication_date
            else:
                # Put undated mentions at the end
                return "9999-12-31"

        sorted_mentions = sorted(all_mentions, key=sort_key)

        # Calculate date range
        date_range = None
        dated_mentions = [m for m in sorted_mentions if m.publication_date]
        if dated_mentions:
            earliest = dated_mentions[0].publication_date
            latest = dated_mentions[-1].publication_date
            date_range = {"earliest": earliest, "latest": latest}

        # Prepare organizations list
        orgs_list = [
            {"vat_id": vat_id, "name": name} for vat_id, name in organizations_involved
        ]

        return TimelineResponse(
            success=True,
            innovation_id=innovation_record["innovation_id"],
            canonical_name=innovation_record["canonical_name"],
            aggregated_description=innovation_record["aggregated_description"],
            total_mentions=len(sorted_mentions),
            date_range=date_range,
            timeline=sorted_mentions,
            organizations_involved=orgs_list,
        )


# Run the server
if __name__ == "__main__":
    logger.info("🚀 Starting VTT Innovation Entity Server with Azure OpenAI")
    mcp.run(transport="stdio")
