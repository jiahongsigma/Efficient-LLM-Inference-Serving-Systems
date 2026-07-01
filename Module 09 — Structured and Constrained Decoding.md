# Module 9 — Structured / Constrained Decoding

*Prerequisites: Module 4 (continuous batching — constraints are per-request, per-step), Module 8 (speculative decoding — another decode-step modification, which constrained decoding interacts with), Module 3 (the quality-measurement discipline). Modules 1–8 made decode fast and big; this module changes *what the decode step is allowed to emit*. It is the first half of Part V — controlling the model's output — and the prerequisite for tool-use (Module 10), because a tool call is structured output.*

---

## In plain English

**Why this matters.** Your application needs the model's output in an exact shape — valid JSON, a value from a fixed list, a SQL query the database will actually accept. Left alone, a model gives you *mostly*-valid output, and "mostly" is exactly what breaks the program that has to read it.

**What this module gives you.** How to *guarantee* the output obeys a schema or grammar (not hope it does), what that guarantee costs in serving speed, and the trap that a perfectly-valid answer can still be a *wrong* answer.

**How it works (the intuition).** At each step the model picks the next token from the whole vocabulary; constrained decoding simply crosses off every choice that would break the format, so whatever it picks stays valid by construction. The catch: forcing the format from the very first token can make the model dumber — so let it reason first, then constrain only the final answer.

---

## 9.1 The problem: applications need structure, models emit tokens

Downstream systems need outputs in a precise format — JSON matching a schema, a value from an enum, a regex, or a full grammar (e.g. valid SQL). A model left to free-sample produces *mostly* valid JSON and *occasionally* a missing brace, a hallucinated field, or prose wrapped around the object — and "occasionally invalid" breaks the consumer. The question is how to **guarantee** valid structure, and what guaranteeing it costs in serving.

This is a decode-step intervention, like Module 8's speculation — but where speculation modified the step for *speed*, constrained decoding modifies it for *structural correctness*. Both live at the logit/sampling stage.

---

## 9.2 The mechanism: logit masking by a grammar state machine

The model normally samples from the full vocabulary. Constrained decoding **restricts the choice to tokens that keep the output valid**:

1. Compile the constraint (JSON schema → regex → context-free grammar) into a **finite-state machine** (FSM) — or a pushdown automaton for context-free grammars.
2. At each decode step, the FSM's **current state** defines the set of **allowed next tokens**.
3. **Mask** every disallowed token's logit to `−∞` before sampling.
4. Sample; advance the FSM by the chosen token; repeat.

Because every emitted token keeps the FSM in a valid state, the output is **valid by construction** — not "usually valid." The guarantee is structural and absolute.

---

## 9.3 The hard part: tokens don't align with the grammar

The grammar is defined over **characters**; the model emits **tokens** (subword units). A single token may cross a grammar boundary, or be valid only as a prefix. So the FSM cannot be built over characters and applied naively — it must be built over the **token vocabulary**: for each FSM state, precompute *which of the ~100K+ tokens are allowed*. This **token–vocabulary alignment** is the core engineering problem of constrained decoding, and where implementations differ:

- **Outlines** builds an FSM-indexed map from states to allowed token sets.
- **XGrammar** (2024) uses a byte-level, compressed-mask approach with precomputed adaptive masks — the current fast engine.
- **llguidance**, **lm-format-enforcer** — other production implementations.

---

## 9.4 What it costs in serving

The cost has two distinct components, and the serving-systems angle is in how they interact with the engine:

- **Compile cost** — building the FSM / token index for a grammar. For a complex CFG this can be slow. It is **amortized by caching** the compiled grammar across requests that share it (a fixed output schema is compiled once, reused forever).
- **Per-step mask cost** — applying the mask at every decode step. With precomputed token masks this is cheap; XGrammar drives it near zero by overlapping mask computation with GPU compute. But it is non-zero and adds to TPOT (Module 4's decode cost).

**The batching interaction (Module 4).** In a continuous batch, different requests carry **different grammars in different FSM states**, so the mask is *per-sequence, per-step*. The engine tracks each sequence's grammar state and applies its own mask — mixing constrained and unconstrained requests in one batch. This is why constrained decoding is an engine feature, not a client-side wrapper.

**The speculative-decoding interaction (Module 8).** If you speculate, the **draft tokens must also satisfy the grammar** — an unconstrained draft proposes tokens the constraint will reject, collapsing the acceptance rate `α`. So the draft path must be grammar-aware for speculation and constrained decoding to compose. Two decode-step modifications that must be co-designed.

---

## 9.5 What it does *not* give you, and the quality trap

The guarantee is **syntactic, not semantic.** Constrained decoding ensures the output *parses* and *matches the schema*; it does **not** ensure the *values are correct* — a schema-valid JSON can still contain a hallucinated number. Validity is necessary, not sufficient.

Worse, and the consistency point of this module: **constraining can degrade quality.** Forcing the model onto a grammar path it would not naturally take — e.g. demanding raw JSON from the first token — denies it the chance to "think" in prose before committing, and can hurt reasoning-heavy answers. The standard mitigation is **reason-then-constrain**: let the model produce free-form reasoning, then constrain only the final structured span (or use a two-call pattern). The quality effect must be *measured* (Module 3/Module 12's quality-aware discipline) — adherence alone is a misleading success metric, because you can hit 100% adherence while the answers got worse.

---

## 9.6 Where the frontier is now

- **XGrammar** (2024) is the current fast structured-generation engine (compressed/adaptive token masks, near-zero per-step overhead), integrated into vLLM and SGLang; SGLang's structured generation and vLLM's guided decoding make this **a first-class engine feature**, not a bolt-on.
- The **quality-degradation finding** and the **reason-then-constrain** pattern are now standard practice.
- Constrained decoding is the **substrate for reliable tool calling and agents** (Module 10) — the reason this module precedes it.

---

## 9.7 The picture to carry forward

- Applications need guaranteed structure; free sampling gives "usually valid," which breaks consumers.
- **Grammar → FSM over the token vocabulary → per-step logit masking** yields output that is **valid by construction**.
- The **token–vocabulary alignment** is the hard part; the cost is **compile (amortize by caching) + per-step (engine-applied, per-sequence)**.
- It **interacts** with batching (per-sequence masks) and speculation (the draft must be grammar-aware).
- It guarantees **syntax, not semantics**, and can **hurt quality** — so measure quality, not just adherence; prefer **reason-then-constrain**.

---

## Lab 9 — Guarantee the structure, then check what it cost

**Context — what this builds on (checklist).** *Reuse:* `common/eval`'s `score_json_schema` (the adherence scorer) and a reasoning task scorer; `common/bench` for the per-step overhead; Module 4's batching. *Test the new insight:* constrained decoding guarantees structure but costs per-step overhead and can hurt quality — not a restated throughput run. *Exercise prior quantities:* Module 4 (batched constrained requests), Module 8 (constrained + speculation), Module 3/12 (quality measurement). *Frontier:* XGrammar vs a naive FSM. *Consistency:* measure the **quality effect**, not adherence alone.

1. **The guarantee (the new insight).** Generate JSON-schema output with and without constrained decoding on the same prompts; score adherence with `score_json_schema`. Unconstrained: < 100% (parse failures, schema violations, prose wrappers); constrained: 100% by construction. *Confirms: §9.2 — validity by construction.*

2. **The two costs (the serving angle).** Measure the **compile cost** (first request with a fresh grammar) and the **per-step TPOT overhead** (constrained vs unconstrained decode); show that **caching the compiled grammar** amortizes the compile cost across reuse. *Confirms: §9.4.*

3. **The quality trap (the consistency centerpiece).** On a reasoning task that also needs structured output, compare (a) forcing JSON from the first token vs (b) **reason-in-prose-then-constrain** the final JSON; score answer quality both ways. Show that constraining too early can *lower quality at 100% adherence*. *Confirms: §9.5 — syntax ≠ semantics; adherence is not quality.*

4. **(Frontier) engine comparison.** XGrammar vs a naive/FSM approach — the per-step overhead difference. *Confirms: §9.6.*

5. **(Interactions) batching + speculation.** Run a batch mixing different grammars (and unconstrained requests); then, if available, constrained + speculative decoding, and observe the acceptance-rate effect of an unconstrained vs grammar-aware draft. *Confirms: §9.4 — the Module 4/8 interactions.*

**Deliverable:** the adherence table (constrained vs not, → 100%); the compile + per-step overhead measurement with the caching amortization; the **quality comparison** (forced vs reason-then-constrain) showing adherence and quality are different axes; and the engine overhead comparison. **Mastery test — defend in one sentence each:** *how constrained decoding guarantees validity (FSM logit masking); the token–vocabulary alignment problem; the two cost components and how to amortize the first; why it guarantees syntax but not semantics and can hurt quality; and how it interacts with batching and speculation.* *Feeds:* Module 10 (tool calls are constrained structured output) and Module 12 (measuring the quality effect, not just adherence).

**Reading:** the **Outlines** paper (FSM-based constrained generation) and **XGrammar** (2024, fast structured generation). Background: vLLM guided-decoding and SGLang structured-generation documentation.