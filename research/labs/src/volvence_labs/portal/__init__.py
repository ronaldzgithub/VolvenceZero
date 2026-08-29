"""Read-only Research Lab aggregation and local serving surface."""

from .collector import PraxistStatusError, ResearchLabCollector, command_status_loader
from .models import (
    ArtifactRef,
    AuthoritySnapshot,
    EvidenceSnapshot,
    HealthStatus,
    LifecycleSnapshot,
    LifecycleStage,
    PortalWarning,
    PraxistRunSnapshot,
    ResearchLabItem,
    ResearchLabSnapshot,
    ResearchLabSummary,
    SourceHealth,
    WarningSeverity,
)
from .server import create_server, serve

__all__ = [
    "ArtifactRef",
    "AuthoritySnapshot",
    "EvidenceSnapshot",
    "HealthStatus",
    "LifecycleSnapshot",
    "LifecycleStage",
    "PortalWarning",
    "PraxistRunSnapshot",
    "PraxistStatusError",
    "ResearchLabCollector",
    "ResearchLabItem",
    "ResearchLabSnapshot",
    "ResearchLabSummary",
    "SourceHealth",
    "WarningSeverity",
    "command_status_loader",
    "create_server",
    "serve",
]
