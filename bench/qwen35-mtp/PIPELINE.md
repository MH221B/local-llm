# Qwen3.5-4B-MTP pipeline tracking

High-level status tracker. Not a spec. Validation evidence lives in `results.md`.

The updated pipeline adds **KV Quantization-Aware Training (KV-QAT)** to Stage 1
fine-tuning, so the student model can deploy a native 4-bit KV cache at 128k/256k context
lengths without reasoning degradation.

**Status legend:** ✅ validated · ⚠️ needs rework · ⏳ not started · ❌ refuted · 🟡 partial

**Overall:** feasible with corrections. The risky runtime assumptions (MTP speculation,
quantized KV on Vulkan, long-context recall under q4_0) are validated locally. The
performance claims are refuted. As of 2026-09-28.

---

## Phase 1: Dataset generation (teacher inference)

*Objective: Synthesize high-quality reasoning and uncensored trajectories on Colab or a local GPU.*

1. **Load Teacher:** Run `mradermacher/MiMo-Ornith-9B-AGSI-i1-GGUF` (e.g., `Q4_K_M` shard) via `llama.cpp` (`llama-server`) or `vLLM` in 4-bit/FP8 precision.
2. **Execute Ingestion:** Feed 10,000–20,000 targeted prompts covering reasoning chains, complex logic, and uncensored tasks.
3. **Format & Filter:**
    - Export outputs to JSONL format with structured `prompt` and `response` keys matching your student's chat template.
    - Strip empty outputs, looping degeneration patterns, and malformed tags.

> **Status:** ⏳ not started
> **Notes:** teacher exists (qwen35 9B VLM). vLLM 4-bit/FP8 support for qwen35 is the main
> risk; llama.cpp fallback works but is throughput-limited. The ~3 h estimate assumes
> large-batch vLLM. Long reasoning outputs (2–8k tokens) make 6–12 h more realistic.

---

## Phase 2A: Stage 1 training, student bf16 LoRA fine-tuning + KV-QAT

*Objective: Update the core transformer weights while conditioning Key/Value projections to tolerate 4-bit compression.*

**Before any training:** score the Phase 1 vision holdout (`eval/vision.jsonl`) on the current model plus the original `mmproj`, and the refusal slice, to establish the pre-training baselines that Phase 1 §11.5's deltas are measured against. Without these the post-training numbers are uninterpretable.

1. **Load Student Base:** Load the base `safetensors` checkpoint of the 4B student (e.g., `RBergBauer/Qwen3.5-4B-MTP-Heretic` or upstream base) in **bf16** and attach LoRA adapters via PEFT or TRL. **Do not use NF4 / QLoRA** — see the status note below.
2. **Inject KV-QAT Forward Hooks:**
    - Wrap or hook the `k_proj` and `v_proj` layers in each attention block with a **FakeQuantize (INT4 / STE)** operation.
    - During the forward pass, simulated quantization rounds and clamps the projected Key and Value activations into 16 discrete symmetric/asymmetric bins before computing scaled dot-product attention ($Q \cdot K^T$).
    - Straight-Through Estimators (STE) pass standard gradients backward, so the attention weights learn to eliminate outlier spikes.
3. **Attach LoRA Projections:**

    ```
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    ```

4. **Train on Standard Context:**
    - Train for 2–3 epochs using context windows of **2,048 to 4,096 tokens** (do not train directly at 128k; RoPE preserves length extrapolation natively).
5. **Merge Weights:**
    - Bake the LoRA adapters back into the 16-bit base weights:

        ```
        model.save_pretrained_merged("qwen-4b-stage1-merged", tokenizer, save_method="merged_16bit")
        ```

> **Status:** ⚠️ needs rework (four corrections)
> 1. **NF4 QLoRA is discouraged by Unsloth for Qwen3.5** ("higher than normal quantization
>    differences"). Use **bf16 LoRA** (~10 GB VRAM, fits an L4).
> 2. **`target_modules` is wrong for this architecture.** Only the **8 full-attention layers**
>    (of 32) have `q_proj/k_proj/v_proj/o_proj`; the other 24 are Gated DeltaNet. Use
>    `target_modules="all-linear"` or the real Qwen3.5 module names.
> 3. **KV-QAT scope + scheme are wrong.** Fake-quant **only the 8 full-attention layers'**
>    `k_proj`/`v_proj` outputs, **post-RoPE for K**, matched to llama.cpp's **32-element block
>    q4_0** scheme (fp16 scale per block), not a global "16-bin" clamp. A global clamp is a
>    different transform and will not transfer to runtime q4_0. GDN recurrent state is
>    separate and constant-size (~50 MiB).
> 4. **"RoPE preserves length extrapolation natively" is false.** The base is natively 262k
>    because it was *trained* that way. Short-context SFT can regress long-range behaviour.
>    Needle-test at 128k/256k after merge.
> **Merge risk:** confirm the merge preserves the vision tensors **and** all 15 MTP tensors
> (`block_count = 33`); heretic's own merge silently dropped the MTP head.
> **Validated locally:** the MTP head is present in the quant and fully used at runtime when
> MTP is enabled.

---

## Phase 2B: Stage 2 training, MTP draft head alignment

*Objective: Align auxiliary draft prediction heads to match the fine-tuned base distribution.*

1. **Freeze Core Network:** Load the merged Stage 1 model and freeze all standard transformer layers.
2. **Extract & Cache Representations:**
    - Run a forward pass over 2,000–5,000 domain prompts to extract hidden states ($h_t$) from the final layer alongside target token embeddings.
3. **Train MTP Draft Heads:**
    - Train strictly the auxiliary speculative projection layers (`mtp.transformer_block` and draft projection heads) to predict token $t+2$ given $h_t$ and $y_{t+1}$.
    - Training takes ~15–30 minutes on an L4 GPU due to the small parameter footprint (~200M–300M parameters).
4. **Consolidate:** Save the aligned MTP weights into the unified checkpoint directory (`./qwen-4b-final-hf`).

> **Status:** ⏳ not started. Premise is sound.
> **Notes:** MTP head is 1 layer / 15 tensors (`blk.32.nextn.*`). Parameter count is
> ~100M, not 200–300M. 15–30 min on an L4 is plausible. Alignment is only needed once
> Stage 1 changes the base distribution.

---

## Phase 3: High-precision GGUF conversion and imatrix quantization

*Objective: Generate the importance matrix and compile the final deployable GGUF.*

1. **Convert to BF16 GGUF:**

    ```
    python llama.cpp/convert_hf_to_gguf.py ./qwen-4b-final-hf \
      --outtype bf16 \
      --outfile qwen-4b-mtp-bf16.gguf
    ```

2. **Compute the Importance Matrix:**
    - Run `llama-imatrix` with 100–200 representative chunks from your teacher dataset:

        ```
        ./llama.cpp/llama-imatrix \
          -m qwen-4b-mtp-bf16.gguf \
          -f calibration_dataset.txt \
          -o imatrix.dat \
          -ngl 99
        ```

3. **Quantize Weights:**

    ```
    ./llama.cpp/llama-quantize \
      --imatrix imatrix.dat \
      qwen-4b-mtp-bf16.gguf \
      qwen-4b-mtp-Q4_K_M.gguf \
      Q4_K_M
    ```

> **Status:** ⏳ not started
> **Notes:** imatrix improves **weights only**. It does nothing for the KV cache (that is
> Phase 2A). The model is a VLM + MTP: the mmproj must be produced separately, and the 15
> MTP tensors must survive convert *and* quantize (verify `block_count = 33` per step).

---

## Phase 4: Local deployment on AMD Strix Point (Ryzen AI 300)

*Objective: Run high-throughput inference at 128k/256k context inside 8 GB VGM.*

1. **Memory Configuration:**
    - Open the BIOS or AMD Adrenalin software and set **Variable Graphics Memory (VGM) to 8 GB**.
2. **Launch with Speculative Decoding and Quantized KV Cache:**
    - Run `llama-server` compiled with Vulkan acceleration:

        ```
        ./llama-server \
          -m qwen-4b-mtp-Q4_K_M.gguf \
          -c 131072 \
          -ngl 99 \
          --flash-attn \
          --cache-type-k q4_0 \
          --cache-type-v q4_0 \
          --spec-type draft \
          --spec-draft-max 2
        ```

    - **Memory Allocation at 128k Context:**
        - Model Weights (`Q4_K_M`): ~2.8 GB
        - Quantized KV Cache (`q4_0` via KV-QAT): ~2.4 GB
        - Scratch / Vulkan Context Buffers: ~0.6 GB
        - **Total VRAM:** ~5.8 GB, inside the 8 GB VGM ceiling.
    - **Performance:** **~70–90+ tokens/second** sustained with MTP speculation, no precision collapse over extended documents.

> **Status:** 🟡 partially validated. Flags and numbers need correction
>
> **Command fixes (build 11228):**
> - `--flash-attn` -> **`-fa on`** (on/off/auto)
> - `--spec-type draft` -> **`--spec-type draft-mtp`**
> - `--spec-draft-max 2` -> **`--spec-draft-n-max 2`**
> - add **`-np 1`** (MTP requires it)
>
> **Corrected memory @128k (measured):**
>
> | Component | Spec | Measured |
> |---|---:|---:|
> | Weights | 2.8 GB | 2.86 GiB |
> | q4_0 KV @128k | 2.4 GB | **1.125 GiB** (3.56x vs f16) |
> | Buffers | 0.6 GB | ~0.4 GiB |
> | **Total** | ~5.8 GB | **4.37 GiB @128k** (5.49 GiB @256k) |
>
> - **VGM 8 GB is wrong for this box.** It has ~19.8 GiB. Memory is not the constraint; bandwidth is.
> - **~70–90 t/s is refuted.** Measured: **29.6 t/s** short-ctx baseline, **~40 t/s** with MTP
>   (n-max 2, 1.35x), **19.4 t/s @57k**, **14.2 t/s @114k**.
> - **Vision + MTP coexist** on build 11228: both `--mmproj` and `--spec-type draft-mtp` load
>   in one server, and MTP speculates (acceptance 0.71-0.89) while answering image requests.
>   The upstream "`--mmproj` not supported with MTP" limitation does not apply here.
>   Confirmed at 8k ctx; `-np 1` required.
>
> **Validated:** q4_0 KV runs on Vulkan (`K (q4_0)` applied, not ignored); MTP speculation works;
> needle @128k with q4_0 **FOUND** (single needle, 50% depth, not yet parity proof vs f16).
> **Batch settings (settled):** keep defaults `-b 2048 -ub 512`. Larger `-ub` degrades prefill
> (-2% at 1024, -11% at 2048); larger `-b` does not change prefill. Prefill measured:
> 659 t/s @4k, 533 t/s @16k, ~170 t/s @114k. (An unconfirmed ~+10% tg128 appears for
> `-b > 2048`; not adopted.)

---

## Updated Google Colab Pro compute ledger (~100 units)

| **Operation** | **Hardware Tier** | **Time Spent** | **Colab Units Used** | **Notes** |
| --- | --- | --- | --- | --- |
| **Phase 1: 9B Inference** | L4 GPU (~3 units/hr) | ~3.0 hrs | **~9 units** | Batch inference with vLLM / llama.cpp |
| **Phase 2A: 4B bf16 LoRA + KV-QAT** | L4 GPU (~3 units/hr) | ~4.0 hrs | **~12 units** | Slight overhead (~15%) from FakeQuant hooks |
| **Phase 2B: MTP Head Alignment** | L4 GPU (~3 units/hr) | ~0.5 hrs | **~1.5 units** | Fast convergence on cached hidden states |
| **Phase 3: Conversion & Imatrix** | L4 / High-RAM CPU | ~1.0 hr | **~3 units** | Matrix evaluation and export |
| **Total Pipeline Spend** |  | **~8.5 hours** | **~25.5 units** | **~74.5 units left over** |

> **Status:** ⏳ rate/hours need verifying
> - Unit arithmetic is internally consistent (~25.5 of 100), but **Phase 1 and 2A hours are
>   optimistic** (especially 2A at bf16 LoRA). Budget **~50–70 units**.
> - **Verify the L4 unit rate and availability.** L4 may not be offered on base Colab Pro.

---

## Open questions

1. Vision + MTP coexistence: **confirmed** on build 11228 (both load and speculate). Open:
   whether it holds at 128k context.
2. KV-quality parity (f16 control @128k, q4_0 @ 90% depth): **deferred, not needed.** The
   single 128k needle FOUND at 50% depth is accepted as sufficient for a fun/learning build.
   Residual: parity is demonstrated once, not proven.
3. Prefill tuning: `-b`/`-ub` settled (defaults optimal). Open: Performance-power-mode rerun
   and a ROCm-vs-Vulkan prefill A/B.
4. Motivation is **learning, not necessity.** The base is already long-context, reasoning-
   capable, uncensored, and MTP-equipped, so Stage 1-3 are done to learn the pipeline (KV-QAT,
   MTP alignment, GGUF conversion), not because the base can't already do the job.

## Artifacts

- `bench/qwen35-mtp/results.md`: validation results
- `bench/qwen35-mtp/needle.ps1`: long-context recall probe
- `bench/qwen35-mtp/prefill-ub-sweep.txt`: prefill `-ub`/`-b` sweep
- `bench/qwen35-mtp/prefill-b-sweep.txt`: prefill `-b` sweep at `-ub 512`
- `models/qwen35-mtp/`: model + mmproj
