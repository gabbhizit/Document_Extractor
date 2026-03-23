# Document Extractor — Phase 1 MVP

End-to-end pipeline for extracting structured data from Indian documents:
**PAN Card · Aadhaar Card · Study Certificates**

---

## Project Structure

```
Document_Extractor/
├── app/
│   ├── main.py              # FastAPI app entry point + logging setup
│   ├── routes.py            # POST /api/v1/extract  |  GET /api/v1/health
│   ├── services/
│   │   ├── ocr.py           # PaddleOCR singleton
│   │   ├── classifier.py    # Keyword-based document type classifier
│   │   ├── extractor.py     # OpenAI LLM structured extraction
│   │   ├── validator.py     # Field validation + confidence scoring
│   │   └── cost_tracker.py  # Token cost in USD + INR
│   ├── utils/
│   │   ├── pdf_parser.py    # PDF → image conversion (pdf2image)
│   │   └── image_utils.py   # Preprocessing + rotation + skew correction
│   └── ui/
│       └── streamlit_app.py # Streamlit demo UI
├── .env                     # Local env vars (never commit)
├── .env.example             # Safe template — commit this
├── Procfile                 # Render / Railway deployment
├── requirements.txt
└── README.md
```

---

## Local Setup

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd Document_Extractor

python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 2. Install system dependencies

**macOS:**
```bash
brew install poppler
```

**Linux (Render / Ubuntu):**
```bash
apt-get install -y poppler-utils libgl1
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Open .env and set your OPENAI_API_KEY
```

---

## Running Locally

### FastAPI backend (port 8000)

```bash
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/v1/health`

### Streamlit UI (port 8501)

Open a second terminal:

```bash
source venv/bin/activate
streamlit run app/ui/streamlit_app.py
```

UI: `http://localhost:8501`

---

## Sample API Request

```bash
curl -X POST "http://localhost:8000/api/v1/extract" \
  -F "file=@/path/to/pan_card.jpg"
```

### Sample Response

```json
{
  "document_type": "PAN",
  "extracted_data": {
    "name": "RAHUL SHARMA",
    "pan_number": "ABCDE1234F",
    "date_of_birth": "01/01/1990"
  },
  "validation": {
    "is_valid": true,
    "errors": []
  },
  "confidence": 0.9,
  "cost": {
    "input_tokens": 312,
    "output_tokens": 45,
    "cost_usd": 0.000074,
    "cost_inr": 0.00703
  },
  "ocr_text": "INCOME TAX DEPARTMENT\n...",
  "processing_time_seconds": 3.142
}
```

---

## Deploying on Render

### Backend (FastAPI)

1. Push repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Or use the `Procfile` (Render picks it up automatically)
4. Add environment variables (see table below)

> **Note:** Render's free tier runs on Linux. Add a build command to install poppler:
> Under **Build Command** use:
> `apt-get install -y poppler-utils libgl1 && pip install -r requirements.txt`

### Streamlit UI (optional separate service)

1. Create a second **Web Service** on Render
2. **Start Command:** `streamlit run app/ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
3. Add env var: `API_BASE_URL=https://your-fastapi-service.onrender.com/api/v1`

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | Your OpenAI API key |
| `OPENAI_MODEL` | ❌ | `gpt-4o-mini` | OpenAI model (e.g. `gpt-4o`) |
| `INPUT_TOKEN_COST_USD` | ❌ | `0.00015` | Cost per 1K input tokens |
| `OUTPUT_TOKEN_COST_USD` | ❌ | `0.0006` | Cost per 1K output tokens |
| `USD_TO_INR` | ❌ | `83.5` | USD → INR conversion rate |
| `SKIP_ORIENTATION_CORRECTION` | ❌ | `false` | Set `true` to disable rotation/skew correction |
| `API_BASE_URL` | ❌ | `http://localhost:8000/api/v1` | Backend URL for Streamlit UI |

---

## Supported Document Types

| Type | Detection Keywords |
|---|---|
| PAN Card | `INCOME TAX DEPARTMENT`, `PERMANENT ACCOUNT NUMBER` |
| Aadhaar Card | `GOVERNMENT OF INDIA`, `UIDAI`, or 12-digit number pattern |
| Study Certificate | 2+ of: `CERTIFICATE`, `SCHOOL`, `COLLEGE`, `UNIVERSITY`, `CBSE`, `SSC`, `HSC`, etc. |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/extract` | Upload and extract document |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |
