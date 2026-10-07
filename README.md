# ProcureAI

**AI-Powered Vendor Selection & Procurement Decision-Support Application**

ProcureAI is a Streamlit-based academic prototype that combines deterministic multi-criteria vendor scoring with OpenRouter-generated explanations.

## Core principle

The application separates **decision calculation** from **generative AI**:

- Python calculates normalized criterion scores, weighted contributions and vendor rankings.
- OpenRouter explains the calculated recommendation and trade-offs.
- OpenRouter does not calculate or modify the ranking.

## Features

- Synthetic 20-vendor dataset
- CSV/XLSX upload
- Dataset validation
- Five decision criteria
- User-defined weights
- Deterministic weighted scoring
- Vendor ranking
- Score contribution analysis
- Interactive Plotly charts
- Vendor detail view
- What-if/sensitivity analysis
- OpenRouter explanation
- Graceful OpenRouter failure handling
- CSV result download
- Unit tests
- Responsible-AI methodology page

## Technology stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Open Router API
- pytest

## Project structure

```text
ProcureAI/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── vendors.csv
├── prompts/
│   └── procurement_prompt.txt
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── validation.py
│   ├── scoring.py
│   └── ai_engine.py
└── tests/
    ├── __init__.py
    ├── test_scoring.py
    ├── test_validation.py
    └── test_ai_failure.py
```

## Local setup

### 1. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure OpenRouter

Copy `.env.example` to `.env` and add your OpenRouter API key:

```text
OPENROUTER_API_KEY=your_key_here
```

Never commit `.env`.

### 4. Run

```bash
streamlit run app.py
```

The application also works without a OpenRouter key. The deterministic vendor evaluation remains available and the AI section displays a graceful fallback message.

## Scoring methodology

Default weights:

| Criterion | Weight | Direction |
|---|---:|---|
| Cost | 30% | Lower is better |
| Quality | 30% | Higher is better |
| Delivery | 20% | Lower lead time + higher on-time delivery |
| Reliability | 15% | Higher is better |
| Sustainability | 5% | Higher is better |

All criteria are normalized to 0–100.

Delivery combines:

- 40% inverse lead-time score
- 60% on-time delivery score

The final score is the weighted sum of the five criterion scores.

## AI methodology

Open Router receives structured calculated results, including:

- recommended vendor
- runner-up
- criterion scores
- weighted contributions
- selected weights

The model is instructed not to invent facts or alter the ranking.

## Testing

Run:

```bash
pytest -q
```

## Streamlit Community Cloud deployment

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Select the GitHub repository.
4. Set the main file to `app.py`.
5. Add the following secret:

```toml
OPENROUTER_API_KEY = "your_actual_key"
OPENROUTER_MODEL = "your_actual_model"
```

6. Deploy.

Do not put the API key into the GitHub repository.

## Responsible AI

ProcureAI is a decision-support prototype, not an autonomous procurement system.

Users should verify:

- supplier information
- contract terms
- financial stability
- legal/compliance status
- operational capacity
- geopolitical risks
- strategic supplier considerations

The application should not autonomously select, contract or onboard a supplier.

## Academic positioning

The project demonstrates:

- business problem framing
- analytics
- multi-criteria decision support
- generative AI
- explainability
- sensitivity analysis
- data validation
- failure handling
- responsible AI
- human-in-the-loop decision making
