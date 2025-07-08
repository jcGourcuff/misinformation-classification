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

# synthetix data

- example dead serious:
    - "Human activities have unequivocally increased well-mixed greenhouse gas concentrations since 1750, with CO2 reaching 410 ppm, CH4 1866 ppb, and N2O 332 ppb in 2019. Each of the last four decades has been successively warmer than any preceding decade since 1850, with a global surface temperature increase of 1.09°C in 2011-2020 compared to 1850-1900."
    - "Since 1950, global land precipitation has likely increased, particularly since the 1980s, with human influence contributing to these changes and the poleward shift of mid-latitude storm tracks in both hemispheres."

- example fuzzy:
    - "Oh, the ice is melting because of us humans—glaciers are shrinking, Arctic ice is vanishing, and even the Greenland Ice Sheet is dripping away, but don't worry about Antarctica, it's just chilling with no significant trend!"