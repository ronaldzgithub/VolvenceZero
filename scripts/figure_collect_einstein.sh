#!/usr/bin/env bash
# Wave K — pilot real Einstein corpus collection.
#
# End-to-end driver: enqueue reviewer-staged seeds, drive the L0
# crawler, re-clean every raw doc to v1, run all 7 L2 verifiers,
# and (optionally) compile a verified curated bundle.
#
# Steps:
#   1. enqueue-batch  — read seeds.jsonl and queue every URL
#   2. run            — drive scheduler with 0.5 rps + burst 5
#   3. re-clean-all   — produce CleanedDocument vN for every raw
#   4. run-batch      — write 7-axis ledger entries per anchor
#   5. (optional, $REQUIRE_VERIFY=1) bake-bundle --corpus-mode curated
#                     compile a real bundle through the OFFLINE gate
#
# Default output directories (gitignored unless overridden):
#   data/figure_corpus/        L0 / L1 / L2 store roots (shared)
#   data/figure_bundles/       persisted bundles
#   data/figure_audit/         bundle bake audit log
#
# Environment overrides:
#   RUN_ID        — crawl run identifier (default: einstein-2026Q2)
#   SEEDS_FILE    — path to seeds JSONL
#   CORPUS_ROOT   — L0/L1/L2 shared root (default: data/figure_corpus)
#   BUNDLE_ROOT   — persisted bundle root (default: data/figure_bundles)
#   AUDIT_ROOT    — audit log root (default: data/figure_audit)
#   MAX_PAGES     — figure_crawl run --max-pages (default: 30)
#   RATE_RPS      — figure_crawl run --rate-rps (default: 0.5)
#   BURST         — figure_crawl run --burst (default: 5)
#   METADATA_FILE — curator-staged CuratedSourceMetadata JSONL for step 5
#                   (default: <SEEDS_DIR>/curated_metadata.jsonl when present)
#   FIGURE_CONTEXT_FILE — per-figure context JSON for step 4 (optional)
#   REQUIRE_VERIFY — set to 1 to run step 5 with --require-verification-pass
#
# Exit codes mirror the underlying CLIs (0 = ok, 2 = gate block, 3 = io/schema).

set -euo pipefail

RUN_ID="${RUN_ID:-einstein-2026Q2}"
SEEDS_FILE="${SEEDS_FILE:-packages/lifeform-domain-figure/data/seeds/einstein-2026Q2.jsonl}"
CORPUS_ROOT="${CORPUS_ROOT:-data/figure_corpus}"
BUNDLE_ROOT="${BUNDLE_ROOT:-data/figure_bundles}"
AUDIT_ROOT="${AUDIT_ROOT:-data/figure_audit}"
MAX_PAGES="${MAX_PAGES:-30}"
RATE_RPS="${RATE_RPS:-0.5}"
BURST="${BURST:-5}"
METADATA_FILE="${METADATA_FILE:-}"
FIGURE_CONTEXT_FILE="${FIGURE_CONTEXT_FILE:-}"
REQUIRE_VERIFY="${REQUIRE_VERIFY:-0}"

mkdir -p "${CORPUS_ROOT}" "${BUNDLE_ROOT}" "${AUDIT_ROOT}"

CRAWL_CLI=(python packages/lifeform-domain-figure/scripts/figure_crawl.py)
CLEAN_CLI=(python packages/lifeform-domain-figure/scripts/figure_clean.py)
VERIFY_CLI=(python packages/lifeform-domain-figure/scripts/figure_verify.py)
BAKE_CLI=(python -m lifeform_domain_figure.cli)

echo "============================================================"
echo " Wave K pilot Einstein corpus collection"
echo " run_id        = ${RUN_ID}"
echo " seeds_file    = ${SEEDS_FILE}"
echo " corpus_root   = ${CORPUS_ROOT}"
echo " bundle_root   = ${BUNDLE_ROOT}"
echo " audit_root    = ${AUDIT_ROOT}"
echo " max_pages     = ${MAX_PAGES}, rate_rps=${RATE_RPS}, burst=${BURST}"
echo "============================================================"

echo ""
echo "[1/5] enqueue-batch ${SEEDS_FILE} -> ${CORPUS_ROOT}/crawl/${RUN_ID}/"
"${CRAWL_CLI[@]}" enqueue-batch \
    --root "${CORPUS_ROOT}" \
    --run-id "${RUN_ID}" \
    --requests-file "${SEEDS_FILE}"

echo ""
echo "[2/5] run scheduler (rate=${RATE_RPS} rps, burst=${BURST}, max-pages=${MAX_PAGES})"
"${CRAWL_CLI[@]}" run \
    --root "${CORPUS_ROOT}" \
    --run-id "${RUN_ID}" \
    --cleaning-root "${CORPUS_ROOT}" \
    --rate-rps "${RATE_RPS}" \
    --burst "${BURST}" \
    --max-pages "${MAX_PAGES}"

echo ""
echo "[3/5] re-clean-all under ${CORPUS_ROOT}"
"${CLEAN_CLI[@]}" re-clean-all --root "${CORPUS_ROOT}"

echo ""
if [ -n "${PROVENANCE_FILE:-}" ] && [ -f "${PROVENANCE_FILE}" ]; then
    echo "[4/5] verify run-batch (provenance=${PROVENANCE_FILE})"
    VERIFY_ARGS=(
        run-batch
        --root "${CORPUS_ROOT}"
        --provenance-file "${PROVENANCE_FILE}"
        --metadata-mode offline
    )
    if [ -n "${FIGURE_CONTEXT_FILE}" ] && [ -f "${FIGURE_CONTEXT_FILE}" ]; then
        VERIFY_ARGS+=(--figure-context-file "${FIGURE_CONTEXT_FILE}")
    fi
    "${VERIFY_CLI[@]}" "${VERIFY_ARGS[@]}"
else
    echo "[4/5] skipped — set PROVENANCE_FILE=... to run figure_verify run-batch"
fi

echo ""
if [ -n "${METADATA_FILE}" ] && [ -f "${METADATA_FILE}" ]; then
    echo "[5/5] bake-bundle --corpus-mode curated (metadata=${METADATA_FILE})"
    BAKE_ARGS=(
        --bundle-root "${BUNDLE_ROOT}"
        --audit-root "${AUDIT_ROOT}"
        bake-bundle
        --figure einstein
        --corpus-mode curated
        --cleaning-root "${CORPUS_ROOT}"
        --curated-metadata-file "${METADATA_FILE}"
    )
    if [ "${REQUIRE_VERIFY}" = "1" ]; then
        BAKE_ARGS+=(--verification-root "${CORPUS_ROOT}" --require-verification-pass)
    fi
    "${BAKE_CLI[@]}" "${BAKE_ARGS[@]}"
else
    echo "[5/5] skipped — set METADATA_FILE=<curator JSONL> to compile a curated bundle"
fi

echo ""
echo "============================================================"
echo " collection driver complete"
echo " inspect crawl results: ${CORPUS_ROOT}/crawl/${RUN_ID}/results.jsonl"
echo " inspect cleaned text:  ${CORPUS_ROOT}/cleaned/<sha>/v1/text.txt"
echo " inspect verifier log:  ${CORPUS_ROOT}/verification/<sha>/checks.jsonl"
echo "============================================================"
