# P1c v1 aborted after strict lineage rejection

- protocol_id: `d03552af8abb8a31adf0a8424b1f4098c43d4a77918bea3264c64eb24ec203e7`
- baseline_attestation_id: `004cfd15696da8aab8c7934da151756d5a0b52c76268901bb59f5efedd498ab3`
- source P1b report: `647130c49c09a13ecf26db8cac5c1fcacde1a4b02574abe083c9ccac970f7aec`
- terminal checkpoint stage: `p1b_complete`
- authoritative P1c report: **none**

The real Qwen2.5-3B run completed all generations: Gate 0 was 24/24 valid at
10/24 accuracy, and all three P1b contextual arms were 8/8 correct with
pair-flip rate 1.0. The P1c assessor nevertheless rejected the run before
publishing a verdict because the v1 protocol over-bound a fresh
`RelationshipP1ContextBundle.artifact_id`. That artifact includes volatile
MemoryStore entry UUIDs and therefore cannot be equal across independent
runs, even when every evaluated model-input byte is equal.

The source P1 report also marked machinery false because one repo-heldout RAG
pair retrieved the same ordinary background records for both users. Those
heldout contexts were not consumed by this train/validation P1b run; all four
evaluated pairs had distinct contextual bytes and complete scope isolation.
Treating a non-consumed heldout retrieval collision as an evaluated-surface
failure made the gate depend on an unrelated RAG ordering.

No prompt, parser, compiler, model output, generation setting, threshold, or
evaluation label was changed. Protocol v2 replaces the volatile bundle id
with a deterministic hash of the actual train/validation context bytes plus
the frozen background/RAG configs, while scope isolation continues to cover
all users. The v1 artifacts remain immutable diagnostic evidence and must not
be cited as a P1c qualification verdict.
