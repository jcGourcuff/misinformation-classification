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

# First Binary classificatrion

## ministral 3b
- 'not_relevant' sub label removed
- missinformation sub labels merged as one class 'missinformation'
- Binary calssification accuate statement vs missinformation
- result from conufusion matrix: 
    - good precision on missinformation
    - ~ random guess on accurate statements
- focus on misclassified accurate statements
    - personae plays a role but little
    - huge impact of emotion: delusional > sarcastic >> angry >> rest 

- example misclassified misinformation:
    - Atmospheric CO2 is not a pollutant. It is the single most important chemical constituent of the cycles of plant and animal life
    - Earth’s temperature naturally warms and cools cyclically throughout time, and warmer periods are typically more conducive to human thriving than colder periods.
    - Human flourishing depends on energy use, often in invisible and unnoticed ways. A reliable energy supply can feed the hungry, take care of the sick, and ensure a prosperous economic future for society.  It promotes freedom and independence around the world and right here at home,
                    accurate statement  misinformation
accurate statement                 323             242
misinformation                      66            4404
                    Precision  Recall  F1-Score
accurate statement      83.03   57.17     67.72
misinformation          94.79   98.52     96.62
Average                 88.91   77.84     82.17

## ministral 8b
- way better recall on accurate statements
- but much worse on all other metrics
- accurate statement  FPs only on delusional / sarcastic -> confirms that it is about tone
                    accurate statement  misinformation
accurate statement                 518              47
misinformation                     826            3644
                    Precision  Recall  F1-Score
accurate statement      38.54   91.68     54.27
misinformation          98.73   81.52     89.30
Average                 68.64   86.60     71.78

## mistral small
- way better recall on accurate statements
- Same problem as ministral 8b but less pronounced

                    accurate statement  misinformation
accurate statement                 521              44
misinformation                     457            4013
                    Precision  Recall  F1-Score
accurate statement      53.27   92.21     67.53
misinformation          98.92   89.78     94.13
Average                 76.10   91.00     80.83

- misnformatioon classified as accurate examples:
   - Solar, wind and hydropower need help. They can’t produce energy that is available around the clock.
   - In particular, they helped spread a false narrative that arson — and not climate change — was largely to blame for the fires. ‘Bushfires: Firebugs fuelling crisis as national arson arrest toll hits 183,’ read one headline in The Australian, on January 8, 2020. Picked up by Donald Trump, the story was then repeated to millions of Americans by Fox News, also controlled by the Murdoch family.
   - For the past 4567 million years, the sun and the Earth's orbit have driven climate change cycles.

# Split misinformation into sub classes

## ministral 3b

                               Precision  Recall  F1-Score
accurate statement                 97.65   14.64     25.46
fossil fuels needed                73.66   52.80     61.51
not bad                            25.19   76.62     37.91
not happening                      49.29   84.19     62.18
not human                          57.93   58.35     58.14
proponents biased                  86.26   43.41     57.76
science unreliable                 55.28   64.75     59.64
solutions harmful unnecessary      73.04   19.28     30.51
Average                            64.79   51.76     49.14

## ministral 8b

                               Precision  Recall  F1-Score
accurate statement                 84.34   54.14     65.95
fossil fuels needed                76.75   61.19     68.09
not bad                            51.83   40.52     45.48
not happening                      54.27   78.11     64.04
not human                          79.12   30.81     44.35
proponents biased                  90.00   39.18     54.59
science unreliable                 36.97   88.62     52.17
solutions harmful unnecessary      81.12   35.58     49.46
Average                            69.30   53.52     55.52

## mistral small

                               Precision  Recall  F1-Score
accurate statement                 77.13   86.24     81.43
fossil fuels needed                64.90   81.47     72.25
not bad                            66.77   55.84     60.82
not happening                      88.14   65.27     75.00
not human                          72.39   66.29     69.21
proponents biased                  71.59   80.67     75.86
science unreliable                 74.78   53.00     62.03
solutions harmful unnecessary      59.80   76.20     67.01
Average                            71.94   70.62     70.45



# What could have been done
- Better use/prior research on emotions
- Reformulate mis information to aligne tones
- use confidence score 
- reuse uploaded datasets