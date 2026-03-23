"""FastAPI application entry point."""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.routes import router

load_dotenv()

# ── Structured logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Document Extractor API",
    description="Extract structured data from Indian documents: PAN, Aadhaar, Study Certificates",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def on_startup():
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    logger.info("Document Extractor API started — model: %s", model)


@app.get("/", tags=["Root"])
async def root():
    return {"message": "Document Extractor API is running. Visit /docs for API reference."}
