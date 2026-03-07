"""
Estate Agents Demo — AI Listing Copywriter API
Myridian (www.myridian.co.uk)

FastAPI backend that proxies Anthropic API calls with streaming SSE.
Handles rate limiting, generation counting, and CORS.
"""

import json
import os
import time
from collections import defaultdict
from typing import Literal

import anthropic
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse

load_dotenv()

# ============================================================
# App
# ============================================================

app = FastAPI(title="Estate Agents Demo API", version="1.0.0")

CORS_WHITELIST = [
    "https://www.myridian.co.uk",
    "https://myridian.co.uk",
    "https://myridian-website.onrender.com",
    "https://estate-agents-demo.onrender.com",
    "http://localhost:8080",
    "http://localhost:8001",
    "http://localhost:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_WHITELIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Anthropic Client
# ============================================================

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
client = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None

SYSTEM_PROMPT = (
    "You are a professional UK estate agency copywriter specialising in "
    "London residential property. Write compelling, accurate property "
    "descriptions for the address and features provided.\n\n"
    "Tone: professional, warm, no cliches (avoid 'stunning', 'boasting', "
    "'nestled', 'tucked away', 'sought-after').\n\n"
    "Match the requested style:\n"
    "- Portal listing: concise, 150-200 words, factual with personality\n"
    "- Premium brochure: elevated and detailed, 250-300 words, lifestyle-focused\n"
    "- Social post: punchy and emoji-friendly, 80-100 words, hook-first\n\n"
    "Always start with a strong opening line that references the location "
    "or lifestyle. Output only the description — no intro, no commentary, "
    "no labels."
)

STYLE_MAP = {
    "portal": "Portal listing",
    "brochure": "Premium brochure",
    "social": "Social post",
}

# ============================================================
# Rate Limiting (in-memory sliding window)
# ============================================================

rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_MAX = 10
RATE_LIMIT_WINDOW = 3600  # 1 hour


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(ip: str) -> bool:
    now = time.time()
    timestamps = rate_limit_store[ip]
    rate_limit_store[ip] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
    if len(rate_limit_store[ip]) >= RATE_LIMIT_MAX:
        return False
    rate_limit_store[ip].append(now)
    return True


# ============================================================
# Request Schema
# ============================================================


class GenerateRequest(BaseModel):
    address: str
    features: str = ""
    style: Literal["portal", "brochure", "social"] = "portal"


# ============================================================
# Endpoints
# ============================================================


@app.get("/")
def health():
    return {"status": "running", "service": "Estate Agents Demo API"}


@app.get("/api/health")
def detailed_health():
    return {
        "status": "running",
        "service": "Estate Agents Demo API",
        "anthropic_configured": bool(ANTHROPIC_KEY),
    }


@app.post("/api/generate")
async def generate(request: Request, body: GenerateRequest):
    # Validate
    if not body.address.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Property address is required."},
        )

    if not client:
        return JSONResponse(
            status_code=503,
            content={"error": "AI service not configured."},
        )

    # Rate limit
    ip = get_client_ip(request)
    if not check_rate_limit(ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Please try again in a few minutes."},
        )

    # Read generation count from cookie
    current_count = int(request.cookies.get("listing_demo_count", "0"))
    new_count = current_count + 1

    # Build user message
    user_message = (
        f"Property: {body.address}\n"
        f"Features: {body.features or 'None specified'}\n"
        f"Style: {STYLE_MAP[body.style]}"
    )

    # Stream generator
    async def stream():
        start_time = time.time()
        word_count = 0
        full_text = ""

        try:
            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as s:
                for text in s.text_stream:
                    full_text += text
                    yield f"event: token\ndata: {json.dumps({'text': text})}\n\n"

            elapsed_ms = int((time.time() - start_time) * 1000)
            word_count = len(full_text.split())
            yield f"event: done\ndata: {json.dumps({'word_count': word_count, 'generation_time_ms': elapsed_ms})}\n\n"

        except anthropic.RateLimitError:
            yield f"event: error\ndata: {json.dumps({'message': 'AI service is busy. Please try again in a moment.'})}\n\n"
        except anthropic.AuthenticationError:
            yield f"event: error\ndata: {json.dumps({'message': 'AI service authentication error. Please contact support.'})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': f'Generation failed: {str(e)[:100]}'})}\n\n"

    response = StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

    response.set_cookie(
        key="listing_demo_count",
        value=str(new_count),
        max_age=30 * 24 * 3600,
        httponly=False,
        samesite="lax",
        secure=False,  # Allow on localhost; Render handles HTTPS
    )

    return response


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
