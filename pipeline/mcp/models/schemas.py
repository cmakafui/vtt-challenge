#!/usr/bin/env python
# /// script
# dependencies = [
#   "pydantic",
# ]
# ///

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class MentionRecord(BaseModel):
    """Model representing an innovation mention with full provenance."""

    mention_unique_id: str
    original_name: str
    source_url: Optional[str] = None
    source_doc_id: str
    dataset_origin: str
    publication_date: Optional[str] = None
    role_in_mention: Optional[str] = None
    relationship_description_in_mention: Optional[str] = None
    original_text: Optional[str] = None
    confidence: float = 1.0


class Innovation(BaseModel):
    """Model representing a canonical innovation entity."""

    innovation_id: str
    canonical_name: str
    aggregated_description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    labels: Optional[List[str]] = None


class InnovationResponse(BaseModel):
    """Response model for innovation operations."""

    success: bool
    error: Optional[str] = None
    innovation_id: Optional[str] = None
    canonical_name: Optional[str] = None
    aggregated_description: Optional[str] = None
    created: Optional[bool] = None
    similarity_matches: Optional[List[Dict[str, Any]]] = None


class OrganizationResponse(BaseModel):
    """Response model for organization operations."""

    success: bool
    error: Optional[str] = None
    vat_id: Optional[str] = None
    canonical_name: Optional[str] = None
    aliases: Optional[List[str]] = None
    found: Optional[bool] = None


class InvolvementResponse(BaseModel):
    """Response model for involvement operations."""

    success: bool
    error: Optional[str] = None
    fingerprint: Optional[str] = None
    org_vat_id: Optional[str] = None
    innovation_id: Optional[str] = None
    org_name: Optional[str] = None
    innovation_name: Optional[str] = None
    primary_role: Optional[str] = None
    status: Optional[Literal["created", "updated", "duplicate"]] = None
    mentions_count: Optional[int] = None


class SimilarityResponse(BaseModel):
    """Response model for similarity search."""

    query: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0


class MergeInnovationResponse(BaseModel):
    """Response model for merging innovations."""

    success: bool
    error: Optional[str] = None
    innovation_id: Optional[str] = None
    canonical_name: Optional[str] = None
    merged_from: Optional[str] = None
    merged_into: Optional[str] = None


class MergeOrganizationResponse(BaseModel):
    """Response model for merging organizations."""

    success: bool
    error: Optional[str] = None
    vat_id: Optional[str] = None
    canonical_name: Optional[str] = None
    merged_from: Optional[str] = None
    merged_into: Optional[str] = None


class TimelineMention(BaseModel):
    """Model representing a mention in the timeline."""

    mention_unique_id: str
    publication_date: Optional[str] = None
    original_name: str
    source_url: Optional[str] = None
    source_doc_id: str
    dataset_origin: str
    role_in_mention: Optional[str] = None
    relationship_description_in_mention: Optional[str] = None
    original_text: Optional[str] = None
    confidence: float = 1.0
    organization_vat_id: Optional[str] = None
    organization_name: Optional[str] = None


class TimelineResponse(BaseModel):
    """Response model for innovation timeline."""

    success: bool
    error: Optional[str] = None
    innovation_id: Optional[str] = None
    canonical_name: Optional[str] = None
    aggregated_description: Optional[str] = None
    total_mentions: int = 0
    date_range: Optional[Dict[str, str]] = None
    timeline: List[TimelineMention] = Field(default_factory=list)
    organizations_involved: List[Dict[str, str]] = Field(default_factory=list)
