---
## Token-Oriented Object Notation vs JSON: a benchmark of plain and constrained decoding generation"

Token-Oriented Object Notation [https://github.com/toon-format](https://github.com/toon-format) is a compact, human-readable encoding of the JSON data model that minimizes tokens and makes structure easy for models to follow. It's intended for LLM input as a drop-in, lossless representation of your existing JSON.

While TOON is primarily designed for input, its token efficiency makes it a candidate for LLM output in specific high-volume scenarios. This benchmark compares three generation strategies across 21 models:

1.  **JSON (J)**: Plain generation (Pydantic validation).
2.  **JSON Structured Output (JSO)**: Constrained decoding (`response_format="json_object"` / grammar enforcement).
3.  **TOON (T)**: One-shot in-context learning generation.

### Key findings

*   **Aligned data ("sweet spot"):** TOON excels in tabular and uniform nested structures (e.g., invoices, orders), achieving **90.5%** accuracy in 1-shot tests while offering significant token savings.
*   **Prompt tax:** Unlike JSON, which is native to model training, TOON requires instructional prompting. For short outputs, this overhead reduces efficiency; for larger outputs (batches/logs), the syntax savings amortize the cost.
*   **Structured output trade-off:** Constrained decoding (JSO) acts as a safety net for smaller models (preventing syntax errors) but was found to degrade reasoning/accuracy in some larger models ("structured output paradox").

### Results by data topology

Performance varies significantly based on how well the data aligns with TOON's "uniform array" design.

| Case | Structure | Best format | Observation |
| :--- | :--- | :--- | :--- |
| **Users** | Flat tabular | **JSO** (tokens) / **TOON** (acc) | TOON reached **90.5%** 1-shot accuracy. JSO used fewer tokens (556 vs 840) due to TOON's prompt overhead on small tasks. |
| **Order** | Nested + uniform array | **Mixed** | TOON (74.3%) is competitive with JSON (81.9%), proving effective for standard business documents. |
| **Invoice** | Items & totals | **JSO** | TOON struggled initially (0% 1-shot) requiring repair loops, while JSO enforced strict schemas effectively (95.2%). |
| **Company** | Deep/recursive | **TOON** (final efficiency) | **Anomaly:** TOON failed 1-shot (0%) but repair loops made it the *most* accurate final result (48.6%) with the *lowest* token usage (2567). |

### Full results by model

The following table compares **1-shot accuracy (1-S)**, **final accuracy (Fin)** after repair loops, and the total **token budget (Tok)** required for successful generation.

| Model | J (1-S) | J (Fin) | J (Tok) | JSO (1-S) | JSO (Fin) | JSO (Tok) | T (1-S) | T (Fin) | T (Tok) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NousResearch/Hermes-4-405B** | 92.5% | 92.5% | 3252 | 35.0% | **100%** | 4759 | 50.0% | 60.0% | 4671 |
| **NousResearch/Hermes-4-70B** | 75.0% | 75.0% | 4414 | 37.5% | 75.0% | 5594 | 50.0% | 50.0% | 4738 |
| **PrimeIntellect/INTELLECT-3** | 72.5% | 75.0% | 10682 | 72.5% | 77.5% | 10103 | 40.0% | 65.0% | 13315 |
| **Qwen/Qwen2.5-Coder-7B-fast** | 0.0% | 0.0% | 37705 | 75.0% | 75.0% | 4440 | 27.5% | 27.5% | 32715 |
| **Qwen/Qwen3-235B-A22B-Inst** | **100%** | **100%** | 2772 | **100%** | **100%** | 2772 | 50.0% | **100%** | 4715 |
| **Qwen/Qwen3-235B-A22B-Thk** | 82.5% | 82.5% | 11425 | 87.5% | 97.5% | 7899 | 50.0% | 97.5% | 17457 |
| **Qwen/Qwen3-30B-A3B-Inst** | 75.0% | 75.0% | 4436 | 75.0% | 75.0% | 4436 | 50.0% | 70.0% | 5505 |
| **Qwen/Qwen3-32B** | 75.0% | 77.5% | 10196 | 75.0% | 75.0% | 4120 | 47.5% | 80.0% | 9101 |
| **Qwen/Qwen3-Coder-30B-A3B** | 75.0% | 75.0% | 4206 | 75.0% | 75.0% | 4206 | 50.0% | **100%** | 4719 |
| **Qwen/Qwen3-Coder-480B** | 75.0% | 75.0% | 4462 | 75.0% | 75.0% | 4447 | 50.0% | 75.0% | 4515 |
| **deepseek-ai/DeepSeek-R1** | 55.0% | 70.0% | 13811 | 65.0% | 80.0% | 4149 | 25.0% | 50.0% | 19047 |
| **deepseek-ai/DeepSeek-V3-fast** | 75.0% | **100%** | 3600 | 75.0% | **100%** | 3584 | 25.0% | 80.0% | 4734 |
| **google/gemma-2-2b-it** | 75.0% | **100%** | 4721 | 77.5% | **100\%** | 4566 | 0.0% | 0.0% | 5955 |
| **google/gemma-2-9b-it-fast** | 75.0% | 75.0% | 6086 | 75.0% | 75.0% | 6056 | 50.0% | 75.0% | 5419 |
| **meta-llama/Llama-3.3-70B** | 75.0% | 75.0% | 4551 | 75.0% | 75.0% | 4447 | 50.0% | 50.0% | 5148 |
| **meta-llama/Llama-3.1-8B** | 72.5% | 72.5% | 7235 | 75.0% | 75.0% | 6941 | 22.5\% | 25.0% | 4915 |
| **moonshotai/Kimi-K2-Instruct** | 50.0% | 75.0% | 4284 | 50.0% | 75.0% | 4283 | 50.0\% | **100\%** | 3937 |
| **nvidia/Llama-3_1-Nemotron** | 75.0% | 75.0% | 4426 | 50.0% | 50.0% | 5714 | 50.0% | 82.5% | 4368 |
| **openai/gpt-oss-120b** | **97.5%** | **100%** | 3685 | **100%** | **100%** | 3545 | 50.0% | 87.5% | 8223 |
| **openai/gpt-oss-20b** | 50.0% | 72.5% | 14943 | 50.0% | 67.5% | 15601 | 50.0% | 90.0% | 9678 |
| **zai-org/GLM-4.5** | 75.0% | 87.5% | 9677 | 75.0\% | 92.5\% | 9135 | 27.5\% | 52.5\% | 8110 |

### Observations

**1. The "Structured Output Paradox"**
Constrained decoding is not always superior. For `Hermes-4-405B`, applying constraints dropped 1-shot accuracy from **92.5%** (Plain JSON) to **35.0%** (Structured Output). This suggests that for some high-reasoning models, forcing specific grammar paths can actively interfere with the model's logic capabilities.

**2. Guardrails for smaller models**
Conversely, for smaller models like `Qwen/Qwen2.5-Coder-7B-fast`, structured output is essential. It raised performance from a catastrophic **0%** (Plain JSON) to a viable **75%**.

**3. TOON repair potential**
While TOON often has lower initial 1-shot accuracy due to the novelty of the format, several models (`Qwen/Qwen3-Coder-30B`, `Kimi-K2-Instruct`, `Qwen/Qwen3-235B`) achieved **100% final accuracy** after repair loops. This indicates that while the format may be unfamiliar initially, the error messages provided by the TOON CLI are highly effective for self-correction.

**4. Token efficiency scaling**
In cases like `Qwen3-235B-A22B-Inst`, TOON consumed significantly more tokens (~4700) than JSON (~2700). This confirms the "prompt tax" hypothesis: for short tasks, the instructional overhead outweighs the syntax savings. TOON becomes efficient primarily in high-volume generation where the output length justifies the system prompt.

### Analysis & recommendations

1.  **Aligned data streams:** Use TOON generation for **SQL dumps, logs, and transactional documents**. The token savings on high-volume, uniform data outweigh the prompt overhead.
2.  **Avoid deep nesting:** For deeply nested or recursive state trees (like DOMs), stick to **JSON** or **JSO**. TOON's indentation tracking is less robust for these structures in one-shot generation.
3.  **Repair loops:** TOON generation benefits disproportionately from repair loops (feeding errors back to context), often correcting format issues that initial constrained decoding cannot fix.