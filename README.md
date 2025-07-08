# Fake News Detection & Source Attribution


# Use Case Description

Dataset: [Dataset Card for Climate Disinformation quotes database from QuotaClimat & Data For Good
](https://huggingface.co/datasets/QuotaClimat/frugalaichallenge-text-train#dataset-card-for-climate-disinformation-quotes-database-from-quotaclimat--data-for-good)

Use Case: Use LLM to classify statements or headlines as fake/true and optionally explain why

LLM Application:

Baseline: Prompted truth classification

Improvement: Use RAG to cross-check claim with news articles / Wikipedia

Impact: Huge societal benefit (education, journalism, politics)

Evaluation: Truth classification F1, hallucination score, explainability

Focus on climate change disinformation



# Plan

- Dataset preparation: huggig face dataset + positive examples
    - cite IPCC reports
- Demonstratiuon of sub classification of negative Impact
- Base Model
- Evaluation
- RAG
- Fine Tuning
- Slides

- Truth is irremdiably austere