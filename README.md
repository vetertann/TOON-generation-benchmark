---
## Token-Oriented Object Notation vs JSON: a benchmark of plain and constrained decoding generation

While TOON is primarily designed for input, its token efficiency makes it an attractive candidate for LLM output in high-volume systems. This benchmark compares three generation strategies across 21 models:

1.  **JSON (J)**: Plain JSON generation (Pydantic validation).
2.  **JSON Structured Output (JSO)**: Constrained decoding (`response_format="json_object"` / grammar enforcement).
3.  **TOON (T)**: One-shot in-context learning generation.

### Key findings

*   **The "Sweet Spot" (Aligned Data):** TOON excels in tabular and uniform nested structures (e.g., Invoices, Orders), achieving **90.5%** accuracy in 1-shot tests while offering significant token savings.
*   **The "Prompt Tax":** Unlike JSON, which is native to model training, TOON requires instructional prompting. For short outputs, this overhead reduces efficiency; for larger outputs (batches/logs), the syntax savings amortize the cost.
*   **Structured Output Trade-off:** Constrained decoding (JSO) acts as a safety net for smaller models (preventing syntax errors) but was found to degrade reasoning/accuracy in some larger models ("Structured Output Paradox").

### Results by data topology

Performance varies significantly based on how well the data aligns with TOON's "Uniform Array" design.

| Case | Structure | Best Format | Insight |
| :--- | :--- | :--- | :--- |
| **Users** | Flat Tabular | **JSO** (Tokens) / **TOON** (Acc) | TOON reached **90.5%** 1-shot accuracy. JSO used fewer tokens (556 vs 840) due to TOON's prompt overhead on small tasks. |
| **Order** | Nested + Uniform Array | **Mixed** | TOON (74.3%) is competitive with JSON (81.9%), proving effective for standard business documents. |
| **Invoice** | Items & Totals | **JSO** | TOON struggled initially (0% 1-shot) requiring repair loops, while JSO enforced strict schemas effectively (95.2%). |
| **Company** | Deep/Recursive | **TOON** (Final Efficiency) | **Anomaly:** TOON failed 1-shot (0%) but repair loops made it the *most* accurate final result (48.6%) with the *lowest* token usage (2567). |

### Model performance (Selected)

Comparison of **1-Shot Accuracy (1-S)**, **Final Accuracy (Fin)** after repairs, and **Total Token Budget (Tok)**.

| Model | J (1-S) | JSO (1-S) | T (1-S) | T (Fin) | Insight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen/Qwen3-235B** | **100%** | **100%** | 50.0% | **100%** | Strong reasoning adapts well to TOON repairs. |
| **Hermes-4-405B** | 92.5% | 35.0% | 50.0% | 60.0% | JSO significantly degraded reasoning vs plain JSON. |
| **Qwen/Coder-7B** | 0.0% | **75.0%** | 27.5% | 27.5% | JSO rescued the smaller model from total failure. |
| **GPT-OSS-120b** | 97.5% | **100%** | 50.0% | 87.5% | High baseline performance across all formats. |

###  Analysis & recommendations

1.  **Aligned Data Streams:** Use TOON generation for **SQL dumps, Logs, and Transactional Documents**. The token savings on high-volume, uniform data outweigh the prompt overhead.
2.  **Avoid Deep Nesting:** For deeply nested or recursive state trees (like DOMs), stick to **JSON** or **JSO**. TOON's indentation tracking is less robust for these structures in one-shot generation.
3.  **Repair Loops:** TOON generation benefits disproportionately from repair loops (feeding errors back to context), often correcting format issues that initial constrained decoding cannot fix.