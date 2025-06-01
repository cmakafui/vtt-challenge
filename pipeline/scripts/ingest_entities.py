#!/usr/bin/env python3
"""
Optimized organization ingestion with batch operations for maximum speed.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Any

import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.panel import Panel
from rich.table import Table
from neo4j import GraphDatabase, Driver

console = Console()
MEMGRAPH_URI = "bolt://localhost:7687"
BATCH_SIZE = 100  # Process organizations in batches

app = typer.Typer(help="🚀 High-performance organization ingestion with batching")


def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        console.print(f"❌ Error loading {file_path}: {e}", style="red")
        return None


def extract_unique_org_ids_optimized(duplicates_data: Dict[str, Any]) -> Set[str]:
    """
    Optimized ID extraction using set comprehension and flattening.
    """
    duplicates = duplicates_data.get("duplicates", [])

    # Flatten all org IDs in one go using set comprehension
    org_ids = {
        org_id
        for duplicate_pair in duplicates
        for innovation_key in ["innovation1", "innovation2"]
        if innovation_key in duplicate_pair
        for field in ["associated_orgs", "non_vtt_orgs"]
        for org_id in duplicate_pair[innovation_key].get(field, [])
        if isinstance(duplicate_pair[innovation_key].get(field), list)
    }

    return org_ids


def batch_create_organizations(driver: Driver, org_batch: List[Dict[str, Any]]) -> int:
    """
    Create multiple organizations in a single database transaction.
    This is where the real performance gain comes from!
    """
    if not org_batch:
        return 0

    # Build a single Cypher query that handles multiple organizations
    query = """
    UNWIND $organizations AS org_data
    MERGE (org:Organization {vat_id: org_data.vat_id})
    ON CREATE SET
        org.aliases = org_data.aliases,
        org.source_ids = org_data.source_ids,
        org.canonical_name = org_data.canonical_name,
        org.created_at = datetime()
    ON MATCH SET
        org.aliases = org_data.aliases,
        org.source_ids = org_data.source_ids,
        org.canonical_name = org_data.canonical_name,
        org.updated_at = datetime()
    RETURN count(org) as created_count
    """

    try:
        with driver.session() as session:
            result = session.run(query, organizations=org_batch)
            return result.single()["created_count"]
    except Exception as e:
        console.print(f"❌ Batch creation error: {e}", style="red")
        return 0


def prepare_org_data(org_id: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare organization data for batch insertion."""
    aliases = details.get("alias", [])
    source_ids = details.get("source_id", [])

    # Ensure source_ids is a list
    if not isinstance(source_ids, list):
        source_ids = [source_ids] if source_ids is not None else []

    return {
        "vat_id": org_id,
        "aliases": aliases,
        "source_ids": source_ids,
        "canonical_name": aliases[0] if aliases else org_id,
    }


@app.command()
def ingest(
    glossary_file: Path = typer.Argument(..., help="Path to the JSON glossary file"),
    duplicates_file: Path = typer.Argument(
        ..., help="Path to the duplicates JSON file"
    ),
    uri: str = typer.Option(
        MEMGRAPH_URI, "--uri", "-u", help="Memgraph connection URI"
    ),
    batch_size: int = typer.Option(
        BATCH_SIZE, "--batch-size", "-b", help="Batch size for database operations"
    ),
):
    """
    High-performance organization ingestion using batch operations.
    """

    console.print(
        Panel.fit(
            "⚡ High-Performance Organization Ingestion\n"
            f"Using batch size: {batch_size}",
            style="bold blue",
        )
    )

    # Validate files
    for file_path in [glossary_file, duplicates_file]:
        if not file_path.exists():
            console.print(f"❌ File not found: {file_path}", style="red")
            raise typer.Exit(1)

    # Load data
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}")
    ) as progress:
        progress.add_task("Loading glossary...", total=None)
        glossary_data = load_json_file(glossary_file)
        if not glossary_data:
            raise typer.Exit(1)

        progress.add_task("Loading duplicates...", total=None)
        duplicates_data = load_json_file(duplicates_file)
        if not duplicates_data:
            raise typer.Exit(1)

        progress.add_task("Extracting org IDs...", total=None)
        unique_org_ids = extract_unique_org_ids_optimized(duplicates_data)

    console.print(f"📊 Found {len(unique_org_ids)} unique organizations")
    console.print(f"📚 Glossary contains {len(glossary_data)} entries")

    # Connect to database
    try:
        driver = GraphDatabase.driver(uri, auth=None)
        driver.verify_connectivity()
        console.print(f"✅ Connected to Memgraph", style="green")
    except Exception as e:
        console.print(f"❌ Database connection failed: {e}", style="red")
        raise typer.Exit(1)

    # Prepare batches
    valid_orgs = []
    missing_count = 0

    for org_id in unique_org_ids:
        if org_id in glossary_data:
            org_data = prepare_org_data(org_id, glossary_data[org_id])
            valid_orgs.append(org_data)
        else:
            missing_count += 1

    # Process in batches
    total_created = 0
    num_batches = (len(valid_orgs) + batch_size - 1) // batch_size

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "Creating organizations in batches...", total=num_batches
        )

        for i in range(0, len(valid_orgs), batch_size):
            batch = valid_orgs[i : i + batch_size]
            created_in_batch = batch_create_organizations(driver, batch)
            total_created += created_in_batch
            progress.advance(task)

    driver.close()

    # Results
    results_table = Table(title="⚡ High-Performance Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Count", style="magenta", justify="right")

    results_table.add_row("Organizations Created/Updated", str(total_created))
    results_table.add_row("Missing from Glossary", str(missing_count))
    results_table.add_row("Batch Size Used", str(batch_size))
    results_table.add_row("Number of Batches", str(num_batches))

    console.print(results_table)

    if total_created > 0:
        console.print("🚀 High-speed ingestion completed!", style="bold green")


if __name__ == "__main__":
    app()
