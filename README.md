# Misinformation Classification with Mistral API

## Setup

Have an Mistral API key stored in a file at `./env/MISTRAL_API_KEY`

## How to Use

### Load Data

Load data from both data sources, and parse the IPCC reports with OCR:

```bash
python main.py load
```

### Generate Synthetic Samples

To generate synthetic accurate quote samples using Mistral Large:

```bash
python main.py synthesize
```

### Classification Task

You are now ready to perform classification tasks. 

All tasks are done in batches - so that you must wait a bit bewteen the run and eval steps.

To perform classification tasks, use the `class` command with the following options:

```bash
python main.py class --task <task_type> --stage <stage> [--model <model_name>] [--few-shot] [--eval-set <evaluation_set>]
```

#### Options

- `--task`: Classification task type. Choose from `binary` or `multiclass`.
- `--stage`: Processing stage. Choose from:
  - `build`: Creates necessary files from processed data.
  - `run`: Uploads files to Mistral API and launches inference job.
  - `eval`: Evaluates the results of the inference job and saves metrics.
- `--model`: Model to use. Options include:
  - `ministral-3b-latest`
  - `ministral-8b-latest`
  - `mistral-small-latest`
  - `tuned-1-epoch`
  - `tuned-3-epochs`
  - `tuned-10-epochs`

  Default is `ministral-3b-latest`.
- `--few-shot`: Use few-shot learning. Default is `False`. Note that zero-shot is not available for binary classification.
- `--eval-set`: Evaluation set to use. Choose from:
  - `global`: For all data.
  - `validation`: For the validation set used for fine-tuning.
  
  Default is `global`.

#### Example Usage

```bash
# Model doesn't matter at this stage
python main.py class --task multiclass --stage build --few-shot --eval-set validation
```

```bash
python main.py class --task multiclass --stage run --few-shot --model ministral-3b-latest --eval-set validation
```

```bash
python main.py class --task multiclass --stage eval --few-shot --model ministral-3b-latest --eval-set validation
```

# Data Sources

- [IPCC Reports](https://www.ipcc.ch/reports/)
- [QuotaClimat Dataset](https://huggingface.co/datasets/QuotaClimat/frugalaichallenge-text-train)
