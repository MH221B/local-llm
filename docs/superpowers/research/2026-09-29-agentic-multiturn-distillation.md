# Agentic & multi-turn distillation: current practice

Research note for the Qwen3.5-4B SFT project. Scope: how 2025–2026 practice represents,
generates, and filters multi-turn trajectories for a small open model trained by
rejection-sampled SFT on a larger teacher's text. Read alongside
`docs/superpowers/specs/2026-09-28-qwen35-phase1-dataset-design.md` §5–§10.

Confidence is marked per claim: **[strong]** = stated in a primary source, **[weak]** =
inferred or single-source, **[open]** = practitioners disagree or no source found.

---

## 1. How multi-turn agentic trajectories are represented

**The wire format is a full message list with OpenAI-style roles.** A trajectory is one
record: `system`, then repeating `user → assistant → tool → assistant …`, ending on an
assistant turn. NVIDIA's `Nemotron-SFT-Agentic-v2` card describes its rows as
"Conversations + tool specifications + metadata" in JSONL, with separate `messages`,
`tools`, and `parallel_tool_calls` fields and per-turn token accounting in
`metadata.turn_token_count`. **[strong]**
(<https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2>)

**Roles.** Four roles are in play, not three: `system`, `user`, `assistant`, `tool`.
Assistant turns carry tool calls; `tool` turns carry the result. In the OpenAI schema the
fields are `tool_calls` on the assistant message and `tool_call_id` on the tool message.
`ReTool-SFT-multi-turn` shows exactly this per message: every entry has `role`, `content`,
`name`, `tool_call_id`, `tool_calls` keys. **[strong]**
(<https://huggingface.co/datasets/swordfaith/ReTool-SFT-multi-turn>)

**Two encoding choices exist for the tool call itself.**

1. *Structured fields* — `tool_calls` as a JSON array on the assistant message, rendered
   by the chat template. This is what ReTool, Nemotron, and ToolACE-family data use.
2. *Inline text* — the call expressed as text inside `content`, sometimes in XML-ish tags.
   Nemotron 3 explicitly chose XML-style tags "to reduce character escaping, following the
   observations of GLM-4.5 and Qwen3-Coder". **[strong]**
   (<https://arxiv.org/html/2512.20848v1>, §3.1.1)

Which one is viable depends on your template. Qwen3's template renders
`tool_call.arguments` with a type check (`| tojson` only when it is not already a string)
to avoid double-escaping. **[strong]**
(<https://huggingface.co/blog/qwen-3-chat-template-deep-dive>)

**Reasoning interacts with turns.** Two distinct policies appear:

- *Drop earlier reasoning on a new user turn, keep it across tool steps.* Nemotron 3:
  "Multi-Step: existing reasoning tokens are preserved … Multi-Turn: When a user message is
  introduced, any reasoning from previous turns are dropped." **[strong]**
- *Rolling checkpoint.* Qwen3's template walks the message list backwards to the latest
  non-tool user turn, keeps full ` thinking` blocks after that index, strips earlier ones.
  **[strong]** (same HF blog)

This matters directly: a multi-turn record containing per-turn ` thinking` blocks renders
differently depending on how many turns you keep, and earlier blocks may be silently
pruned by the template. **[strong]**

**Shape summary.**

```text
single-turn (today)          multi-turn trajectory
────────────────────         ──────────────────────────────────────────
system?                      system?
user                         user              ← first user turn
assistant                    assistant  (+ tool_calls)
                             tool        (tool result)
                             assistant
                             user              ← later user turn
                             assistant
```

---

## 2. Loss-masking mechanics in practice

**Only assistant spans get loss.** System, user, and tool tokens are set to `-100`
(PyTorch's `ignore_index`). This is universal across the stacks. **[strong]**
(<https://yonigottesman.github.io/2024/05/13/mask-user-tokens.html>)

**Masking is template-driven, not role-parsed by hand.** TRL's `assistant_only_loss=True`
requires the chat template to emit `{% generation %}` / `{% endgeneration %}` markers
around assistant spans; TRL auto-patches known families such as Qwen3. **[strong]**
(<https://huggingface.co/docs/trl/en/sft_trainer>,
<https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md>)

**Known failure modes, all documented:**

| Failure | Cause | Source |
|---|---|---|
| Empty assistant mask → error | Template lacks `{% generation %}`, so no masks are produced | TRL docs / HF forum |
| Mask becomes empty after truncation | Long multi-turn row truncated before the assistant span | HF forum thread on `assistant_only_loss` deprecation discussion |
| VLM path refuses | `assistant_only_loss` blocked for vision-language models, even on text-only data | HF forum: "SFTTrainer flags blocks assistant_only_loss=True" |
| Loss on wrong tokens | Custom collators that locate spans by string matching rather than template markers | Gottesman post (pre-template approach) |

**[strong]** for the first three; the fourth is the historical motivation for the
template-marker design. (<https://discuss.huggingface.co/t/sfttrainerflags-blocks-assistant-only-loss-true/176210>)

**Do people mask tool results?** In practice yes, because masking is applied to everything
that is not inside a generation span, and tool results are not assistant-written. The tool
result is an *observation* the model conditions on, not something it should learn to emit.
**[weak]** — no source states this as a rule, but it follows from every masking
implementation found, and no dataset card describes training on tool output.

**Two templates worth knowing about.** For Qwen3 multi-turn there are community templates
that differ in which assistant spans get the generation mask: `all_assistant.jinja` masks
every assistant reply, `final_assistant.jinja` masks only the last. **[strong]**
(<https://github.com/HarryMayne/qwen_3_chat_templates>)

**Turn expansion vs. whole conversation.** Unsloth documents both: `conversation_extension`
merges N single-turn rows into one conversation at dataset-prep time, which is turn
concatenation rather than expansion. **[strong]**
(<https://docs.unsloth.ai/basics/chat-templates>) Expansion (one example per assistant
reply) is the older ShareGPT-era pattern and is not what current agentic data uses
**[weak]**.

---

## 3. Teacher-side generation for multi-turn

**The dominant pattern is a simulated environment in the loop with three simulated roles.
** NVIDIA's tool-calling subset: "Each trajectory is created by simulating three roles:
User (provides a goal/task), Agent (plans and interacts via tools), Tool environment
(returns tool outputs to be consumed by the agent)." Three separate LLMs, or one LLM
playing all three. **[strong]**
(<https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2>)

**The tool environment is simulated by a model, not necessarily a real sandbox.** In the
Nemotron conversational-tool-use pipeline the user, the agent, and the tool execution
environment are each "simulated by a language model". Real execution appears where the task
demands it — code interpreters, ARC-AGI grid tools, web search, OpenHands. **[strong]**
(arXiv 2512.20848 §3.1.2; Nemotron-SFT-Agentic-v2 card)

**Trajectory lengths vary enormously by task.** Search trajectories "require ~10–30 search
calls". OpenHands SWE trajectories were collected with maximum turns set to 100 and 500.
**[strong]** (<https://nebius.com/blog/posts/openhands-trajectories-with-qwen3-coder-480b>)

**Reasoning is extracted into a separate field at generation time.** ARC-AGI-v1 data:
"chain-of-thought stored in a dedicated `reasoning_content` field on assistant messages,
tool calls in OpenAI function-calling format, empty or error-laden turns" removed.
Nemotron-SFT-Agentic-v2 records `chat_template_kwargs: {"thinking": true}` and a
`metadata.turn_token_count` list splitting `reasoning` from `content` **per turn**.
**[strong]**
(<https://huggingface.co/datasets/nvidia/Nemotron-SFT-ARC-AGI-v1>,
<https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2>)

That per-turn reasoning/content split is the notable design choice: the record stores the
two streams separately and lets the trainer decide how to render them, rather than baking
tags into one content string.

**Scaling.** Nemotron 3 Super's tool-calling pipeline produced "a dataset of 1.5M diverse
tool-calling trajectories" using DeepSeek-v3.2 and GLM-4.7. Nemotron 3 Nano's agentic SFT
used Qwen3-235B-A22B-Thinking-2507, Qwen3-32B, GPT-OSS-120b, and Qwen3-235B-A22B-Instruct-
2507 across the roles. **[strong]** (arXiv 2604.12374; arXiv 2512.20848 §3.1.2)

---

## 4. Rejection sampling and verification for agentic data

**Two granularities of check, routinely both.** Nemotron 3 Super: "we employ a turn-level
and trajectory-level judge … The turn level judge is also paired with a rule-based
verification for ensuring correctness of tool-calls." **[strong]**
(<https://arxiv.org/html/2604.12374v1>)

**Trajectory level** — did it solve the task? Progressive-difficulty web-agent work:
"retaining only those that conclude with answers consistent with the ground truth". ARC-AGI:
"A run is kept only when the agent's submitted grid exactly matches the puzzle's ground
truth (deterministic rule-based filtering — no LLM judge or scoring is applied)". **[strong]**
(<https://arxiv.org/html/2510.13913v1>,
<https://huggingface.co/datasets/nvidia/Nemotron-SFT-ARC-AGI-v1>)

**Turn level** — was each action valid? Structural validity of the tool call, argument
schema conformance, and action consistency against the agent's stated goal. Nemotron 3 Nano:
"we employ a language model as a judge to evaluate the trajectories, and filter out
trajectories for which the judge considers an action of an entity to be inconsistent with
its goals." The published pipeline stages for the agentic set are
`reasoning_extraction, uuid_addition, uuid_removal, identity_filter, propaganda_filter,
loop_detection, tool_validation, token_counting`. **[strong]**
(Nemotron-SFT-Agentic-v2 card; the `stages` field is visible in the viewer)

**Drop granularity.** Sources describe dropping at trajectory level, and separately
cleaning *turns* — removing "empty or error-laden turns". No source found that rewrites a
failed step and keeps the trajectory. **[strong]** for the two practices, **[open]** on
whether step-level repair is done in practice.

**Difficulty filtering is part of the recipe.** "Select successful trajectories for SFT and
filter for difficulty by dropping scenarios that yield all-success or all-failure outcomes."
That is: keep trajectories from tasks where the model sometimes succeeds, since all-pass and
all-fail carry no signal. **[strong]** (arXiv 2604.12374)

**No oracle, no filter.** Where there is no environment, chat-style agentic data falls back
to a model judge or nothing. Nemotron 3 Nano additionally notes exhaustive-turn verifiers:
"keep samples where all turns pass the respective instruction verifier implementations in
IFEval and IFBench." **[strong]** (arXiv 2512.20848)

---

## 5. Roleplay / chat multi-turn distillation

**Prebuilt chat data is natively multi-turn, and it is used as-is.** SmolTalk defines
`systemchats-30k` and `everyday-conversations` as multi-turn sets; smoltalk2 reports mean
turns per example of **6.27** for `systemchats-30k` and **7.75** for
`everyday-conversations`, against ~2 turns for most other subsets. MagPie-Ultra is
described as "three-turn conversations", built by a two-step prompting method. **[strong]**
(<https://huggingface.co/datasets/HuggingFaceTB/smoltalk2>,
<https://arxiv.org/html/2502.02737v1>)

**LongAlign is the long-context slice, and it is short on turns.** smoltalk2 reports
`LongAlign-64k` at **Avg. # turns = 2** with 16,220 avg context tokens but only 2,135
response tokens. In the original SmolLM2 mix, LongAlign contributed 3.73k samples at
8k–16k tokens. So "long context" here means long *single* turns, not many turns. **[strong]**
(<https://arxiv.org/html/2502.02737v1>, smoltalk2 card)

**Persona and consistency are the stated hard part.** The multi-turn survey is blunt:
"instruction retention and self-consistency remain difficult — models frequently fail to
remember instructions or maintain a coherent narrative over several turns. Preserving
conversational state across turns thus remains an open problem." Role-play SFT work
(CharacterGLM, Humanish-Llama-3.1-8B, Peach-9B-Roleplay) shows role models scoring lower on
open-domain single-turn than general instruct models of similar size, attributing it to
training data that "emphasizes limited role styles and persona consistency rather than
factual correctness and coverage". **[strong]**
(<https://arxiv.org/html/2504.04717v6>,
<https://aclanthology.org/2025.findings-acl.1082.pdf>)

**SFT alone is documented as insufficient for multi-turn robustness.** A widely-cited
practitioner position: SFT "jump-starts the agent with reasonable behavior" and "can teach
non-obvious tool usage patterns when those examples appear in the data", but full multi-turn
RL is what makes behavior "robust and adaptive". The survey likewise notes SFT is the most
widely used technique while listing multi-turn RL and multi-turn DPO as the refinement
stage. **[strong]** (<https://fireworks.ai/blog/best-practices-for-multi-turn-RL>)

**Turn-count defaults where data is synthesized, not collected.** TurnWise builds
"multi-turn conversations of up to eight user turns" for evaluation. Multi-IF-style
generation in smoltalk2 "generate[s] 2 verifiable turns". **[strong]**
(<https://arxiv.org/html/2603.16759v1>, modelscope smoltalk2 page)

---

## 6. Concrete numbers and defaults

| Quantity | Value | Source |
|---|---|---|
| Turns, chat multi-turn sets | 6–8 mean | smoltalk2 |
| Turns, synthesized IF | 2 | smoltalk2 / Multi-IF |
| Turns, eval conversations | up to 8 | TurnWise |
| Turns, search agent | 10–30 tool calls | Nemotron-SFT-Agentic-v2 |
| Turns, SWE agent eval | 100 or 500 max | Nebius OpenHands |
| Tool trajectories, one production set | 1.2M (~707k tool_calling subset) | Nemotron-SFT-Agentic-v2 |
| Tool trajectories, another | 1.5M | Nemotron 3 Super |
| Total agentic SFT storage | ~20 GB | Nemotron-SFT-Agentic-v2 |
| Avg context tokens, agentic tool trace | 6,934 (vs 682 response) | smoltalk2 (`smolagents-toolcalling-traces`) |
| Avg context, long-context slice | 16,220 | smoltalk2 (LongAlign-64k) |

**Share of multi-turn in a mixed corpus.** smoltalk2 reports `smoltalk-everyday-convs` at
2,057 examples (0.06% of examples) and `smoltalk-systemchats` at 27,436 (0.81%), against
`xlam-traces` at 59,962 (1.77%) and multi-turn-reasoning at 28,217 (0.83%). In SmolTalk v1,
outside of MagPie-Ultra (431k) the multi-turn chat sets were small: SystemChats2.0 35.9k,
LongAlign 3.73k, Everyday-Conversations 2.38k samples. **[strong]** — so in a chat-heavy
corpus the multi-turn *chat* share is modest and mostly arrives via the synthetic MagPie
family. **[weak]**

**No source found** that reports a recommended multi-turn percentage of an SFT mix, or a
token budget rule for multi-turn vs single-turn. **[open]**

---

## What this implies for a 4B single-teacher SFT pipeline

Stated as constraints and tradeoffs, not as a recommendation.

**Constraints the sources establish:**

1. A multi-turn record needs a fourth role (`tool`) and two new message fields
   (`tool_calls`, `tool_call_id`). The current `ROLES = ("system", "user", "assistant")`
   and `validate_prompt`'s "last role is user" rule cannot express a trajectory and will
   reject one. **[strong]**

2. Every practical stack masks by chat-template generation markers, not by hand. So the
   render gate (spec §11.2) becomes load-bearing for multi-turn: it must show which spans
   the project's Qwen3.5 template marks, including whether tool results land inside or
   outside a generation span. **[strong]**

3. Per-turn reasoning needs a decision the current verbatim-tags rule does not cover.
   Qwen3's rolling checkpoint silently prunes earlier ` thinking` blocks, and Nemotron 3
   deliberately drops earlier reasoning on a new user turn. Storing one content string per
   turn means the template decides what survives. **[strong]**

4. A single 9B local teacher playing user + agent + tool-environment is the documented
   pattern, and it is cheap relative to RL. But it is a different generation shape from the
   current seeded single completion: N sequential calls per trajectory, with the environment
   simulated. Resume/idempotency across a multi-call trajectory is a new requirement,
   versus the current per-request content hash. **[weak]**

5. Verification splits cleanly into trajectory-level (did it solve it) and turn-level (was
   each call valid), and the sources run both. Unit tests you already have cover the
   trajectory level for code. Turn-level tool-call validity is rule-based and cheap. A model
   judge is what the sources use for action coherence, and it is the only option for
   roleplay. **[strong]**

6. Roleplay multi-turn has no mechanical check even in the literature, and the survey
   states conversational state preservation is unsolved. Any roleplay multi-turn program
   rests on judge scoring plus human audit. **[strong]**

**Tradeoffs the sources expose but do not settle:**

- Whole-conversation records are cheap in tokens and match how chat sets ship; turn
  expansion multiplies token cost by the turn count but makes every example self-contained
  and trivially fits the existing single-completion machinery. **[open]**
- Multi-turn SFT is documented as necessary but not sufficient for robustness; sources
  expect a later RL or DPO stage. A project with no RL stage gets the "reasonable behavior"
  tier, not the robust tier. **[strong]**
- Keeping the trailing-tool-result row shape (a trajectory ending on a `tool` turn) versus
  requiring an assistant turn to finish: sources do not discuss it. **[open]**

**Where sources conflict or are thin:** nobody documents masking policy for tool *results*
explicitly; nobody gives a target multi-turn share for a mixed SFT corpus; nobody reports
step-level repair of failed trajectories; and roleplay multi-turn quality metrics are
judge-based and acknowledged as unreliable.
