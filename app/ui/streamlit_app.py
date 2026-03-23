"""Streamlit demo UI for the Document Extractor API."""

import os
import time

import requests
import streamlit as st

# Allow API base URL to be set via env var for deployed environments
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="Document Extractor",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Indian Document Extractor")
st.caption("Supports PAN Card · Aadhaar Card · Study Certificates")
st.divider()

uploaded_file = st.file_uploader(
    "Upload a document (JPG, PNG, PDF)",
    type=["jpg", "jpeg", "png", "pdf"],
)

if uploaded_file is not None:
    file_type = uploaded_file.type

    # Preview image uploads
    if file_type in ("image/jpeg", "image/png", "image/jpg"):
        st.image(uploaded_file, caption="Uploaded Document", width=700)

    st.divider()
    extract_btn = st.button("🔍 Extract Document", type="primary", use_container_width=True)

    if extract_btn:
        t0 = time.time()
        with st.spinner("Running OCR and extracting data — please wait..."):
            try:
                response = requests.post(
                    f"{API_BASE}/extract",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), file_type)},
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to the API. "
                    "Make sure the FastAPI backend is running and `API_BASE_URL` is correct."
                )
                st.stop()
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The document may be too large or the server is busy.")
                st.stop()
            except requests.exceptions.HTTPError as e:
                detail = "Unknown error"
                try:
                    detail = e.response.json().get("detail", str(e))
                except Exception:
                    detail = str(e)
                st.error(f"❌ API Error: {detail}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                st.stop()

        # ── Results ───────────────────────────────────────────────────────────
        doc_type    = result.get("document_type", "UNKNOWN")
        confidence  = result.get("confidence", 0.0)
        extracted   = result.get("extracted_data", {})
        validation  = result.get("validation", {})
        cost        = result.get("cost", {})
        ocr_text    = result.get("ocr_text", "")
        server_time = result.get("processing_time_seconds")
        client_time = round(time.time() - t0, 2)

        # ── Top metrics row ───────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Document Type", doc_type)
        with col2:
            conf_label = f"{confidence:.0%}"
            delta_color = "normal" if confidence >= 0.7 else "inverse"
            st.metric("Confidence", conf_label)
        with col3:
            display_time = f"{server_time}s" if server_time else f"~{client_time}s"
            st.metric("Processing Time", display_time)

        st.divider()

        # ── Extracted Fields ──────────────────────────────────────────────────
        st.subheader("📋 Extracted Data")
        if extracted:
            cols = st.columns(2)
            for i, (key, value) in enumerate(extracted.items()):
                cols[i % 2].text_input(
                    key.replace("_", " ").title(),
                    value=value or "—",
                    disabled=True,
                )
        else:
            st.warning("No data extracted.")

        st.divider()

        # ── Validation ────────────────────────────────────────────────────────
        st.subheader("✅ Validation")
        if validation.get("is_valid"):
            st.success("All fields validated successfully.")
        else:
            for err in validation.get("errors", []):
                st.error(err)

        st.divider()

        # ── Cost breakdown ────────────────────────────────────────────────────
        st.subheader("💰 API Cost")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Input Tokens",  cost.get("input_tokens", 0))
        c2.metric("Output Tokens", cost.get("output_tokens", 0))
        c3.metric("Cost (USD)",    f"${cost.get('cost_usd', 0.0):.5f}")
        c4.metric("Cost (INR)",    f"₹{cost.get('cost_inr', 0.0):.4f}")

        st.divider()

        # ── Expandable sections ───────────────────────────────────────────────
        with st.expander("🔤 Raw OCR Text"):
            st.text(ocr_text if ocr_text.strip() else "No OCR text available.")

        with st.expander("📦 Full API Response (JSON)"):
            st.json(result)
