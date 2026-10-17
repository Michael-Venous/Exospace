# Model server (B200)

**Owner:** Models

Serves the executor candidate (UI-TARS-1.5-7B) through vLLM's OpenAI-compatible API at `EXECUTOR_BASE_URL`.

Gotcha: UI-TARS coordinates refer to its internally resized image, so map them back to the captured frame before converting to normalized coordinates (PLAN.md §5.2).
