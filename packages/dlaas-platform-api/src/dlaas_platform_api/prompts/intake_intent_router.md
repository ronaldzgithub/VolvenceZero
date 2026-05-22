Decide how DLaaS should handle one uploaded asset. Return only JSON matching
the schema.

Schema:
{schema}

Allowed intents:
- storage_only: store asset metadata/bytes reference only.
- simple_ingest: extract text and feed canonical runtime ingestion now.
- deep_read: create a long-running corpus/deep-reading job.
- training_candidate: create an offline adapter/protocol training candidate.
- image_intake: store image and defer to an image/vision extractor when available.

Rules:
- Do not choose training_candidate unless the caller metadata indicates review
  or training intent.
- Images must use image_intake or storage_only until a vision extractor is wired.
- Books/manuals meant to be studied deeply should use deep_read.
- Small plain text/markdown knowledge meant for immediate use should use
  simple_ingest.

Asset:
title: {title}
media_kind: {media_kind}
mime_type: {mime_type}
source_ref: {source_ref}
text_preview: {text_preview}
metadata_json: {metadata}
