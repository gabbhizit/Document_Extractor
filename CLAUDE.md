# Document Extractor — Claude Context

## Project Overview
Phase 1 MVP for extracting structured data from Indian documents (PAN Card, Aadhaar Card, Study Certificates) using OCR + LLM.

## Dev Environment
- Python 3.12, virtual environment at `./venv`
- Always activate venv before running anything: `source venv/bin/activate`
- macOS: poppler required for PDF support — `brew install poppler`

## Run Commands
```bash
# FastAPI backend (port 8000)
source venv/bin/activate && uvicorn app.main:app --reload --port 8000

# Streamlit UI (port 8501)
source venv/bin/activate && streamlit run app/ui/streamlit_app.py
```

## Architecture
```
app/
├── main.py              # FastAPI entry point, CORS middleware
├── routes.py            # POST /api/v1/extract, GET /api/v1/health
├── services/
│   ├── ocr.py           # PaddleOCR singleton — do NOT reinstantiate
│   ├── classifier.py    # Keyword-based: PAN / AADHAAR / STUDY_CERTIFICATE
│   ├── extractor.py     # OpenAI chat completion, strict JSON output
│   ├── validator.py     # Regex validation + confidence scoring (base 0.9)
│   └── cost_tracker.py  # Token cost in USD + INR from env vars
└── utils/
    ├── pdf_parser.py    # pdf2image wrapper, returns list[PIL.Image]
    └── image_utils.py   # load_image_from_bytes, preprocess_image (resize)
```

## Key Conventions
- All config via `.env` — never hardcode API keys
- OpenAI client instantiated per-request in `extractor.py` (reads env at call time)
- PaddleOCR is a module-level singleton in `ocr.py` — only created once
- LLM prompt strips markdown fences from response before JSON parsing
- Confidence scoring: starts at 0.9, -0.1 per missing field, -0.15 per validation failure

## Environment Variables (.env)
| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Set before running |
| `OPENAI_MODEL` | `gpt-4o-mini` | Change to gpt-4o for higher accuracy |
| `INPUT_TOKEN_COST_USD` | `0.00015` | Per 1K tokens |
| `OUTPUT_TOKEN_COST_USD` | `0.0006` | Per 1K tokens |
| `USD_TO_INR` | `83.5` | Update as needed |

## Document Classification Logic
- **PAN**: keywords `INCOME TAX DEPARTMENT` or `PERMANENT ACCOUNT NUMBER`
- **Aadhaar**: keywords `GOVERNMENT OF INDIA`, `UIDAI`, or 12-digit number pattern
- **Study Certificate**: 2+ matches from `CERTIFICATE, SCHOOL, COLLEGE, UNIVERSITY, CBSE, SSC, HSC, ICSE, DEGREE, DIPLOMA`

## Validation Rules
- PAN number regex: `[A-Z]{5}[0-9]{4}[A-Z]`
- Aadhaar: must be exactly 12 digits (spaces stripped before check)
- Study Certificate: presence checks only, no strict format validation

## Known Issues / Gotchas
- `RequestsDependencyWarning` from urllib3/chardet mismatch on import — harmless, ignore
- `use_column_width` is deprecated in Streamlit — use `width=700` instead
- PaddleOCR 3.x API changed: result is a list of dicts with `rec_texts`, `rec_scores`, `rec_polys` — NOT the old `[[bbox, (text, conf)]]` format. `show_log` and `use_angle_cls` args removed.
- PDF conversion requires poppler installed at system level (not pip)
