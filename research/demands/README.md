# ResearchDemand inbox

This directory contains sealed `forge-volvence-research-demand.v1` artifacts. A Demand is a complete Volvence-owned
research need, not a Codex suggestion and not approval to start Praxist.

Author drafts outside this directory, omit `demand_id`, and preserve an explicit `created_at`. Then seal them with:

```bash
forge research-demand-seal research/demand_drafts/<name>.json --json
```

The sealer computes the canonical identity, validates every evidence/source path, and writes a create-only JSON file here.
The managed Research Lab worker discovers only `OPEN` Demand JSON files; unchanged Demand and source bytes cause no model
call. A resulting TopicProposal remains `UNBOUND` until a named human binds it to an exact registry mapping, after which a
separate A0 approval is still required.

Minimum draft shape:

```json
{
  "schema_version": "forge-volvence-research-demand.v1",
  "claim_id": "claim:<registered-claim>",
  "title": "A concrete, bounded research need",
  "objective": "The falsifiable result Volvence needs",
  "owner": "<registered-task-owner>",
  "capability_axes": ["readable"],
  "need": {
    "current_gap": "What current evidence cannot establish",
    "required_outcome": "What a useful experiment must establish",
    "success_criteria": ["A preregistered measurable success condition"],
    "falsification_criteria": ["A preregistered result that rejects the hypothesis"],
    "protected_boundaries": [
      "Evaluation is not a learning source",
      "No production wiring changes"
    ]
  },
  "evidence": [],
  "discovery": {
    "source_roots": ["research/<bounded-corpus>"],
    "max_source_files": 64,
    "max_source_bytes": 1048576,
    "max_topics": 4
  },
  "routing": {"requested_mapping_id": null},
  "status": "OPEN",
  "authority": {
    "discovery_only": true,
    "human_topic_binding_required": true,
    "human_a0_required": true,
    "research_start_authorized": false,
    "formal_validation_performed": false,
    "production_promotion_authorized": false,
    "runtime_wiring_changed": false,
    "evaluation_is_learning_source": false
  },
  "created_at": "<RFC3339 timestamp>"
}
```
