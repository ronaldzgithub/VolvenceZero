# Figure Corpus Cleaning Pipeline (L1) Spec

> Status: draft
> Last updated: 2026-05-10
> 对应需求: R8（快照优先 / 单一所有者）、R15（迁移可解释性 + 可回滚）

## 要解决的问题

`docs/known-debts.md` debt #28 把 figure vertical 的 webcrawl + 数据清洗 + 多源验证拆成 L0 / L1 / L2 三层。本 spec 只覆盖 **L1 cleaning pipeline**：把一个具体的源字节流（PDF / MediaWiki HTML / Project Gutenberg HTML 或 plain text / Internet Archive OCR JSON）变成 cleaner 版本化、content-addressable、可重跑、可回滚的中性 cleaned text 制品，再桥接到既有的 `CPAEPayload` / `WikisourcePayload` / `GutenbergPayload` / `InternetArchivePayload`。

L0（crawler frontier / robots.txt / 调度）与 L2（多源验证、身份消歧、抽样人审）**不在本 packet 范围**，仍是 follow-up。

为什么需要它：

- 既有 4 个 archive 适配器的 `*Payload.body` 字段假设输入已经是 cleaned plain text。一旦 V2 fetcher 接通真实字节流（PDF / HTML / OCR JSON），这个假设立刻塌陷：要么 V2 fetcher 内部偷做 cleaning（违反单一职责，污染 audit），要么把脏文本灌进 retrieval index。L1 把"字节 → cleaned text"立成独立责任链，让 V2 fetcher 只关心 URL → bytes，cleaner 只关心 bytes → text，bridging 只关心 text → typed payload。
- cleaner 升级（如换更好的 PDF parser）必须能批量重跑而不丢旧版本，否则 audit 不到"这一批 bundle 用的哪个 cleaner 出的"。content-addressable + 版本目录是这条要求的最小落地。

## 关键不变量

1. **Cleaner / parser 不直接产 typed source**。`cleaning/` 子包下的任何模块**禁止** import `FigurePaperSource` / `FigureLetterSource` / `FigureLectureSource` / `FigureNotebookSource`。crossing into typed source 必须经 `cleaning/bridging.py` 的 `cleaned_to_*_payload(...)` + 既有 `*_to_*_source(...)` 二段式（R8 / `ssot-module-boundaries.mdc`）。
2. **Cleaner / parser 不发 HTTP**。`cleaning/` 子包下的任何模块**禁止** import 任何 HTTP 客户端（`requests` / `httpx` / `aiohttp` / `urllib*` / `http.client`）。cleaning 与 fetcher 在职责上完全解耦，wiring 由调用方（curator CLI / 未来 V2 HTTPArchiveFetcher）做。
3. **Raw bytes 是 content-addressable**。`CleaningStore.put_raw(data, ...)` 用 `sha256(data)` 作为唯一 key；同 bytes 第二次 put 短路返回相同 sha，落盘文件不重写。
4. **Cleaner 版本化 = 多版本共存**。`data/cleaned/{raw_sha256}/v{N}/` 是新版本目录，**绝不**覆盖旧版本目录；`store.list_cleaned_versions(raw_sha256)` 返回的元组按数值升序排列。
5. **每个 cleaning op 是 monotonically non-expanding**。`CleaningOpRecord(chars_after > chars_before)` 在 `__post_init__` 直接 raise；orchestrator 按顺序记录每个 op 的字符增减。
6. **Parser self-report 必填**。`RawDocument.parser_version` 形如 `"<parser-id>:<int>"`；`layout_quality` / `ocr_confidence` 在 `[0, 1]`；`raw_sha256` 必须是 64 字符 hex。
7. **Pipeline 升级即 spec 变更**。新增 `CleaningOp` 必须同步 (a) `raw_document.py` enum 加成员；(b) `cleaners/<new>.py` 新文件；(c) `cleaners/__init__.py` 注册新 `CLEANER_PIPELINE_V{N+1}` 与 bump `CURRENT_CLEANER_PIPELINE_VERSION`；(d) 本 spec 加 op 描述。

## Schema

```python
class CleaningOp(str, Enum):
    BOILERPLATE_STRIP = "boilerplate_strip"
    WHITESPACE_NORMALIZE = "whitespace_normalize"
    TYPOGRAPHY_NORMALIZE = "typography_normalize"
    DEDUPE_INTRA_DOC = "dedupe_intra_doc"
    PII_REDACT = "pii_redact"
    PARAGRAPH_NORMALIZE = "paragraph_normalize"

@dataclass(frozen=True)
class CleaningOpRecord:
    op: CleaningOp
    op_version: str        # e.g., "1"
    chars_before: int
    chars_after: int       # invariant: <= chars_before

@dataclass(frozen=True)
class RawDocument:
    text: str
    parser_version: str    # "<parser-id>:<int>", e.g., "cpae-pdf:1"
    layout_quality: float  # [0, 1]
    ocr_confidence: float  # [0, 1]; non-OCR parsers set 1.0
    encoding_detected: str # "utf-8" / "latin-1" / ...
    language_detected: str # ISO-639; "" if unknown
    license_notice: str    # parser-scraped page-level license; "" if absent
    raw_sha256: str        # 64-char hex; content-addressable key

@dataclass(frozen=True)
class CleanedDocument:
    text: str
    raw_sha256: str
    cleaner_pipeline_version: int       # >= 1
    cleaning_log: tuple[CleaningOpRecord, ...]
    parser_version: str                 # passed through from RawDocument
```

## Parsers (4)

| Parser | Source archive | Content-type label | Implementation |
|---|---|---|---|
| `parse_cpae_pdf` | Princeton CPAE | `application/pdf` | `pypdf.PdfReader` page-by-page text extract; first-80-line license-keyword scan; layout_quality from page count + text/byte ratio |
| `parse_wikisource_html` | Wikisource | `text/html; profile=wikisource` (or `text/x-wiki` for raw wikitext) | `bs4` selects `#mw-content-text`; `mwparserfromhell` strips templates; `{{PD-*}}` / `{{license-*}}` / `{{copyright-*}}` templates captured into `license_notice`; `<html lang>` for language |
| `parse_gutenberg` | Project Gutenberg | `text/html; profile=gutenberg` or `text/plain; profile=gutenberg` | HTML path uses `bs4`; both paths carve body around `*** START OF ... ***` / `*** END OF ... ***` markers; pre-START block captured as `license_notice` |
| `parse_archive_org_ocr_json` | Internet Archive | `application/json; profile=archive-org-ocr` | stdlib `json`; concatenates pages with `\f` separator; `ocr_confidence` = mean of page confidences; `metadata.licenseurl` / `.license` captured |

`parse_by_content_type(...)` is a typed dispatcher; unknown content-type labels raise (no silent fallback).

## Cleaner pipeline V1 (6 ops, in order)

| Order | Op | Module | Action | Length effect |
|---|---|---|---|---|
| 1 | `BOILERPLATE_STRIP` | `cleaners/boilerplate.py` | drop bare page-number lines, `Vol. N, p. M` running heads, `[p. N]` markers, form-feeds | strict reduction |
| 2 | `WHITESPACE_NORMALIZE` | `cleaners/whitespace.py` | collapse runs of spaces / tabs to one; collapse 3+ blank lines to 2; trim per-line + document edges | reduction |
| 3 | `TYPOGRAPHY_NORMALIZE` | `cleaners/typography.py` | curly quotes → straight; en/em/minus dashes → ASCII `-`; NBSP → space; rejoin hyphenated line breaks (`experi-\nment` → `experiment`) | length-preserving for char remaps; reduction for rejoin |
| 4 | `DEDUPE_INTRA_DOC` | `cleaners/dedupe.py` | drop verbatim repeated paragraphs (≥ 80 chars) | reduction |
| 5 | `PII_REDACT` | `cleaners/pii.py` | regex-redact emails / phone numbers / credit-card-shaped strings → `[PII]` | reduction (every pattern ≥ 5 chars) |
| 6 | `PARAGRAPH_NORMALIZE` | `cleaners/paragraph.py` | reflow lines: continuation lines (no terminating `.!?:;'")\]}}`) join with space; paragraphs separated by single blank line | length-preserving for joins; reduction for empty-paragraph drop |

`clean_raw_document(raw, *, pipeline_version=CURRENT_CLEANER_PIPELINE_VERSION)` runs the registered pipeline and returns a `CleanedDocument` whose `cleaning_log` records every applied op (even no-op ones, for audit).

## Storage layout

Under the configured `root` (convention: `packages/lifeform-domain-figure/data/`):

```
root/
  raw/
    {sha256}/
      bytes              # exact bytes (immutable; content-addressable)
      sidecar.json       # {source_url, content_type, byte_len, stored_at_ms}
  cleaned/
    {raw_sha256}/
      v{N}/
        text.txt         # CleanedDocument.text
        cleaning_log.json# {raw_sha256, cleaner_pipeline_version, parser_version, ops: [...]}
```

`raw/` and `cleaned/` are listed in the repo `.gitignore`; the parent `data/` ships a `.gitkeep` so the path is stable. Tests use `tmp_path` fixtures and never write to the in-tree `data/`.

## Bridging (cleaned → archive payload)

`cleaning/bridging.py` exposes one pure function per archive:

- `cleaned_to_cpae_payload(cleaned, *, document_id, document_kind, volume, document_number, title, year, language, source_url, sender_id="", recipient_id="", date_iso="") -> CPAEPayload`
- `cleaned_to_wikisource_payload(cleaned, *, page_title, language, source_url, year=None, author_id="", venue_id="", date_iso="", audience="") -> WikisourcePayload`
- `cleaned_to_gutenberg_payload(cleaned, *, ebook_id, title, language, source_url, section_label="", year=None, author_id="") -> GutenbergPayload`
- `cleaned_to_internet_archive_payload(cleaned, *, identifier, title, language, source_url, creator_id="", year=None, venue_id="", date_iso="", audience="") -> InternetArchivePayload`

The curator (or the future V2 fetcher) supplies all archive-specific structural metadata (volume / document_number / page_title / etc.); the parser only sees raw bytes and cannot infer them.

After bridging, the existing `<archive>_to_<source_kind>_source(...)` translators in `lifeform_domain_figure.corpus.archives.*` consume the payload and produce the typed `FigurePaperSource` / `FigureLetterSource` / `FigureLectureSource`.

## CLI

`packages/lifeform-domain-figure/scripts/figure_clean.py` — three subcommands for curators:

```bash
python figure_clean.py clean \
    --root <store-root> --input <file> \
    --content-type 'application/pdf' \
    --source-url 'https://einsteinpapers.press.princeton.edu/vol2-doc/24'

python figure_clean.py re-clean-all \
    --root <store-root> [--pipeline-version N]

python figure_clean.py list-versions --root <store-root>
```

`re-clean-all` iterates every raw under `root/raw/`, parses, runs the requested pipeline (default `CURRENT_CLEANER_PIPELINE_VERSION`), writes the new `v{N}/` directory, and prints a per-raw character-delta vs the highest prior different version (or `no_prior_version_to_diff` when none exists).

## 与 V2 fetcher / L2 verification / L0 crawler 的关系

| Layer | Status | 与 L1 的关系 |
|---|---|---|
| L0 crawler frontier | follow-up packet | crawler 是 fetcher 的 orchestration 层；fetcher 是 crawler 的 leaf node。crawler 输出的字节流喂给 L1 cleaner。 |
| V2 HTTPArchiveFetcher (debt #19 / #26) | follow-up packet | V2 fetcher 拿到 bytes 后调 `CleaningStore.put_raw` + `parse_by_content_type` + `clean_raw_document` + `cleaned_to_*_payload`。fetcher 不做 cleaning（违反 #28 不变量 1）。 |
| L2 verification + audit | follow-up packet | verifier 读 `CleanedDocument` + `RawSidecar` + `cleaning_log.json` 判定 cross-source / identity-disambiguation / authorship 等；输出独立的 `VerificationLedger`。bundle 的 `require_verification_pass` gate 拒收 verdict 不全 PASS 的源。 |

## 接口契约

**Provides**:

- `RawDocument` / `CleanedDocument` / `CleaningOp` / `CleaningOpRecord` neutral schema（figure vertical 内部跨子模块共享，不进 §6 runtime slot 注册表）
- 4 个 parser + dispatcher
- cleaner pipeline orchestrator + `CURRENT_CLEANER_PIPELINE_VERSION`
- `CleaningStore` content-addressable filesystem store
- 4 个 bridging 函数

**Consumes**:

- `pypdf` / `beautifulsoup4` / `mwparserfromhell` / `lxml`（pyproject 已加）
- 既有 `corpus.archives.*` 的 `*Payload` schemas（仅 bridging 引用）

**禁止 import**：

- 任何 `Figure*Source` typed record（违反时 `tests/contracts/test_cleaning_pipeline_versions.py` AST 扫描 FAIL）
- 任何 HTTP 客户端（同上）

## 测试

- `packages/lifeform-domain-figure/tests/test_cleaning_parsers_smoke.py` — 4 parser smoke tests + dispatcher（8 cases）
- `packages/lifeform-domain-figure/tests/test_cleaning_cleaners_smoke.py` — 6 cleaner unit tests + pipeline orchestrator + 5% reduction acceptance（11 cases）
- `packages/lifeform-domain-figure/tests/test_cleaning_store_smoke.py` — content-addressable / sidecar / version isolation / round-trip（8 cases）
- `packages/lifeform-domain-figure/tests/test_cleaning_bridging_smoke.py` — full chain `cleaned -> *Payload -> *_to_*_source -> Figure*Source`（5 cases）
- `tests/contracts/test_cleaning_pipeline_versions.py` — raw_sha 稳定 + v1/v2 目录隔离 + AST 静态扫禁止 typed source / HTTP（7 cases）

## 与其他能力域的关系

| 域 | 关系 |
|---|---|
| Figure Vertical (`docs/specs/figure-vertical.md`) | L1 cleaning 是 figure vertical 的 corpus 数据制备层；V2 fetcher / L2 verification 接通后 cleaning 是它们之间的合约面 |
| Domain Experience Layer | 不直接交互；cleaning 不进运行时 owner |
| Runtime Ingestion | cleaning **之后**的 bridging 输出经 `*_to_*_source` 翻译 → 既有 `ingest_papers` / `ingest_letters` 流入 `lifeform-ingestion` 的 canonical envelope path |

## 变更日志

- 2026-05-10 — 初版落地（debt #28 L1 packet）。Pipeline V1 = 6 ops（boilerplate / whitespace / typography / dedupe / pii / paragraph）；4 parser + 4 bridging + content-addressable store + CLI + 5 个测试套（39 cases，全绿）。
