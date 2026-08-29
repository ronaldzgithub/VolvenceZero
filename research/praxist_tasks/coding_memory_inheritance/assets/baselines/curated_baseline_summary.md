# Curated baseline

The frozen 24-chain Packet 2 v3 report measured `context_token_ratio =
0.10592031659161838`, so the task primary `scaling_margin = 0.10 - ratio` is
`-0.00592031659161838`. The same report retained positive quality gates:
brain-vs-steelman bootstrap lower bound `+0.0047979798` and
brain-vs-stateless lower bound `+0.0075757576`.

Those quality values are provenance and release constraints, not this public
replay evaluator's score. The canonical v2 development replay over chains 0–7
measured ratio `0.1233350275668608`, margin `-0.02333502756686079`, worst-chain
margin `-0.09437921925447607`, recalled/failed selection coverage `1.0 / 1.0`,
selected-line retention `0.972027972027972 / 0.9846153846153847`, and strict
budget pass rate `0.0`.

The baseline therefore exposes two independent repair targets: generic context
keeps the aggregate over the 0.10 bound, and legacy tail-marker truncation both
exceeds the declared character budget and cuts selected owner evidence. The
public replay is suitable for development ranking only; a later sealed coding
hand rerun remains mandatory for quality and release claims.
