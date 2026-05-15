# 在 Git Bash / WSL 里：

# 步骤 1：真 corpus 采集（公共领域，rate-limited）— 约 5-10 min
export REQUIRE_VERIFY=1
export METADATA_FILE=packages/lifeform-domain-figure/data/seeds/einstein-2026Q2.curated_metadata.jsonl
bash scripts/figure_collect_einstein.sh

# 步骤 2：真 PEFT LoRA bake（GPU 必需）— 约 10-20 min
export QWEN_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
export PEFT_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj
export PEFT_DEVICE=cuda
export PEFT_MAX_STEPS=200
bash scripts/figure_bake_einstein_persona_lora.sh

# 步骤 3：4-gate verification harness — 约 5-10 min
export RUNTIME_BACKEND=transformers
export SKIP_BAKE=1
bash scripts/figure_verify_einstein_persona.sh

# 结果落在：
#   artifacts/figure_verify/<run_id>/verdict.json   ← 4-gate 通/不通
#   artifacts/figure_verify/<run_id>/transcript.md  ← 每题 raw/bundle/bundle_lora 三轨答案