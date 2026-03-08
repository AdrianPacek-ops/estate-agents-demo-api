"""
Estate Agents Demo — AI Listing Workflow API
Myridian (www.myridian.co.uk)

FastAPI backend powering the full property listing workflow:
- Photo analysis (Claude Vision)
- Area research (Perplexity API / Claude fallback)
- Multi-variant description generation (Claude SSE streaming)
- Photo processing (Decor8 AI: staging, enhancement, furniture removal)
- Original simple listing generation (backward compatible)
"""

import base64
import json
import os
import time
from collections import defaultdict
from io import BytesIO
from typing import Literal, Optional

import anthropic
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse

load_dotenv()

# ============================================================
# App
# ============================================================

app = FastAPI(title="Estate Agents Demo API", version="2.0.0")

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
# API Clients
# ============================================================

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DECOR8_API_KEY = os.getenv("DECOR8_API_KEY", "")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
DECOR8_BASE_URL = "https://api.decor8.ai"
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None

# ============================================================
# Prompts
# ============================================================

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

PHOTO_ANALYSIS_PROMPT = (
    "You are a professional UK property photographer's assistant. "
    "Analyse each property photo provided and return a JSON array.\n\n"
    "For each photo, return an object with:\n"
    '- "description": 2-3 sentence natural language description of the room/space\n'
    '- "room_type": one of: livingroom, bedroom, kitchen, diningroom, bathroom, '
    "office, openplan, hallway, garden, exterior, balcony, garage\n"
    '- "key_features": array of 3-5 notable features visible in the photo\n\n'
    "Return ONLY valid JSON. No markdown, no commentary.\n"
    "Example: [{\"description\": \"A bright open-plan...\", \"room_type\": \"livingroom\", "
    "\"key_features\": [\"wood flooring\", \"large windows\"]}]"
)

AREA_RESEARCH_PROMPT = (
    "You are a UK estate agent area specialist. For the given property address, "
    "provide detailed local area information that would be useful in a property listing.\n\n"
    "Return ONLY valid JSON with this structure:\n"
    "{\n"
    '  "transport": ["station name (distance, line)", ...],\n'
    '  "schools": ["school name (distance, Ofsted rating)", ...],\n'
    '  "amenities": ["description of nearby amenities", ...],\n'
    '  "lifestyle": "2-3 sentence lifestyle description of the area",\n'
    '  "summary": "One sentence summary of why this location is desirable"\n'
    "}\n\n"
    "Include 3-5 items for transport, 2-3 for schools, 3-5 for amenities. "
    "Be specific with distances and real place names where possible. "
    "Return ONLY the JSON — no markdown, no commentary."
)

DESCRIPTIONS_SYSTEM_PROMPT = (
    "You are a professional UK estate agency copywriter specialising in "
    "London residential property. You have been given:\n"
    "- The property address and key features\n"
    "- AI analysis of the property photos\n"
    "- Local area research\n\n"
    "Write THREE distinct property descriptions. Each should be 180-220 words.\n\n"
    "Variant 1 — PROFESSIONAL: Factual, precise, informative. Focus on specifications "
    "and practical details. Suit a portal listing.\n\n"
    "Variant 2 — LIFESTYLE: Aspirational, evocative, paint-a-picture. Focus on how it "
    "feels to live here. Suit a premium brochure.\n\n"
    "Variant 3 — PUNCHY: Concise, modern, hook-first. Focus on the single most "
    "compelling selling point. Suit social media.\n\n"
    "Separate each variant with exactly this marker on its own line:\n"
    "---VARIANT---\n\n"
    "Tone: professional, warm, no cliches (avoid 'stunning', 'boasting', "
    "'nestled', 'tucked away', 'sought-after').\n"
    "Reference specific details from the photos and area research.\n"
    "Output ONLY the three descriptions with variant markers. No labels, no intro."
)

STYLE_MAP = {
    "portal": "Portal listing",
    "brochure": "Premium brochure",
    "social": "Social post",
}

# ============================================================
# Rate Limiting
# ============================================================

rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_MAX = 10
RATE_LIMIT_WINDOW = 3600

workflow_rate_limit_store: dict[str, list[float]] = defaultdict(list)
WORKFLOW_RATE_LIMIT_MAX = 3
WORKFLOW_RATE_LIMIT_WINDOW = 3600


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


def check_workflow_rate_limit(ip: str) -> bool:
    now = time.time()
    timestamps = workflow_rate_limit_store[ip]
    workflow_rate_limit_store[ip] = [t for t in timestamps if now - t < WORKFLOW_RATE_LIMIT_WINDOW]
    if len(workflow_rate_limit_store[ip]) >= WORKFLOW_RATE_LIMIT_MAX:
        return False
    workflow_rate_limit_store[ip].append(now)
    return True


# ============================================================
# Helpers
# ============================================================


def image_to_data_url(image_bytes: bytes, content_type: str) -> str:
    """Convert image bytes to a base64 data URL."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = content_type or "image/jpeg"
    return f"data:{mime};base64,{b64}"


async def read_upload(upload: UploadFile, max_size: int = 4 * 1024 * 1024) -> bytes:
    """Read and validate an uploaded image file."""
    data = await upload.read()
    if len(data) > max_size:
        raise ValueError(f"Image too large ({len(data)} bytes). Maximum is {max_size // (1024*1024)}MB.")
    ct = upload.content_type or ""
    if not ct.startswith("image/"):
        raise ValueError(f"Invalid file type: {ct}. Please upload a JPEG, PNG, or HEIC image.")
    return data


# ============================================================
# Request Schemas
# ============================================================


class GenerateRequest(BaseModel):
    address: str
    features: str = ""
    style: Literal["portal", "brochure", "social"] = "portal"


class ResearchAreaRequest(BaseModel):
    address: str


class GenerateDescriptionsRequest(BaseModel):
    address: str
    features: str = ""
    photo_analyses: list = []
    area_research: dict = {}


class ProcessPhotoRequest(BaseModel):
    image_data_url: str
    processing_type: Literal["stage", "enhance", "declutter"]
    room_type: str = "livingroom"


# ============================================================
# Health Endpoints
# ============================================================


@app.get("/")
def health():
    return {"status": "running", "service": "Estate Agents Demo API", "version": "2.0.0"}


@app.get("/api/health")
def detailed_health():
    return {
        "status": "running",
        "service": "Estate Agents Demo API",
        "version": "2.0.0",
        "anthropic_configured": bool(ANTHROPIC_KEY),
        "decor8_configured": bool(DECOR8_API_KEY),
        "perplexity_configured": bool(PERPLEXITY_API_KEY),
    }


# ============================================================
# Original Generate Endpoint (backward compatible)
# ============================================================


@app.post("/api/generate")
async def generate(request: Request, body: GenerateRequest):
    if not body.address.strip():
        return JSONResponse(status_code=400, content={"error": "Property address is required."})
    if not claude_client:
        return JSONResponse(status_code=503, content={"error": "AI service not configured."})

    ip = get_client_ip(request)
    if not check_rate_limit(ip):
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded. Please try again in a few minutes."})

    current_count = int(request.cookies.get("listing_demo_count", "0"))
    new_count = current_count + 1

    user_message = (
        f"Property: {body.address}\n"
        f"Features: {body.features or 'None specified'}\n"
        f"Style: {STYLE_MAP[body.style]}"
    )

    async def stream():
        start_time = time.time()
        full_text = ""
        try:
            with claude_client.messages.stream(
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
            yield f"event: error\ndata: {json.dumps({'message': 'AI service authentication error.'})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': f'Generation failed: {str(e)[:100]}'})}\n\n"

    response = StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
    response.set_cookie(key="listing_demo_count", value=str(new_count), max_age=30 * 24 * 3600, httponly=False, samesite="lax", secure=False)
    return response


# ============================================================
# WORKFLOW: Analyse Photos (Claude Vision)
# ============================================================


@app.post("/api/workflow/analyse-photos")
async def analyse_photos(
    request: Request,
    photo_1: UploadFile = File(...),
    photo_2: Optional[UploadFile] = File(None),
    photo_3: Optional[UploadFile] = File(None),
):
    if not claude_client:
        return JSONResponse(status_code=503, content={"error": "AI service not configured."})

    ip = get_client_ip(request)
    if not check_workflow_rate_limit(ip):
        return JSONResponse(status_code=429, content={"error": "Workflow rate limit exceeded. Please try again later."})

    start_time = time.time()

    # Read and validate all uploaded photos
    photos = []
    data_urls = []
    for i, upload in enumerate([photo_1, photo_2, photo_3]):
        if upload is None or upload.filename == "":
            continue
        try:
            img_bytes = await read_upload(upload)
            ct = upload.content_type or "image/jpeg"
            data_url = image_to_data_url(img_bytes, ct)
            photos.append({"index": i, "data_url": data_url, "content_type": ct})
            data_urls.append(data_url)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    if not photos:
        return JSONResponse(status_code=400, content={"error": "At least one photo is required."})

    # Build Claude Vision message with all photos
    content_blocks = []
    for p in photos:
        # Extract base64 from data URL
        b64_data = p["data_url"].split(",", 1)[1]
        media_type = p["content_type"]
        if media_type == "image/heic" or media_type == "image/heif":
            media_type = "image/jpeg"  # Claude doesn't support HEIC directly
        content_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64_data},
        })

    content_blocks.append({
        "type": "text",
        "text": f"Analyse these {len(photos)} property photo(s). Return a JSON array with one object per photo, in order.",
    })

    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=PHOTO_ANALYSIS_PROMPT,
            messages=[{"role": "user", "content": content_blocks}],
        )

        raw_text = response.content[0].text.strip()
        # Try to parse JSON (handle markdown code blocks)
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        analyses = json.loads(raw_text)
        elapsed_ms = int((time.time() - start_time) * 1000)

        return JSONResponse(content={
            "success": True,
            "analyses": analyses,
            "photo_data_urls": data_urls,
            "generation_time_ms": elapsed_ms,
        })

    except json.JSONDecodeError:
        return JSONResponse(status_code=500, content={"error": "Failed to parse photo analysis. Please try again."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Photo analysis failed: {str(e)[:150]}"})


# ============================================================
# WORKFLOW: Research Area (Perplexity API / Claude fallback)
# ============================================================


@app.post("/api/workflow/research-area")
async def research_area(request: Request, body: ResearchAreaRequest):
    if not body.address.strip():
        return JSONResponse(status_code=400, content={"error": "Address is required."})

    start_time = time.time()

    user_prompt = (
        f"Provide detailed local area information for this UK property address: {body.address}\n\n"
        "Include nearby transport links with distances, schools with Ofsted ratings, "
        "local amenities and shopping, and a lifestyle description of the neighbourhood."
    )

    # Try Perplexity first, fall back to Claude
    if PERPLEXITY_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                resp = await http_client.post(
                    f"{PERPLEXITY_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "sonar",
                        "messages": [
                            {"role": "system", "content": AREA_RESEARCH_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "max_tokens": 1500,
                    },
                )

                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data["choices"][0]["message"]["content"].strip()
                    # Parse JSON
                    if raw_text.startswith("```"):
                        raw_text = raw_text.split("```")[1]
                        if raw_text.startswith("json"):
                            raw_text = raw_text[4:]
                        raw_text = raw_text.strip()

                    research = json.loads(raw_text)
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    return JSONResponse(content={
                        "success": True,
                        "research": research,
                        "source": "perplexity",
                        "generation_time_ms": elapsed_ms,
                    })
        except Exception:
            pass  # Fall through to Claude

    # Claude fallback
    if not claude_client:
        return JSONResponse(status_code=503, content={"error": "AI service not configured."})

    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=AREA_RESEARCH_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_text = response.content[0].text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        research = json.loads(raw_text)
        elapsed_ms = int((time.time() - start_time) * 1000)

        return JSONResponse(content={
            "success": True,
            "research": research,
            "source": "claude",
            "generation_time_ms": elapsed_ms,
        })

    except json.JSONDecodeError:
        return JSONResponse(status_code=500, content={"error": "Failed to parse area research. Please try again."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Area research failed: {str(e)[:150]}"})


# ============================================================
# WORKFLOW: Generate Descriptions (Claude SSE — 3 variants)
# ============================================================


@app.post("/api/workflow/generate-descriptions")
async def generate_descriptions(request: Request, body: GenerateDescriptionsRequest):
    if not body.address.strip():
        return JSONResponse(status_code=400, content={"error": "Address is required."})
    if not claude_client:
        return JSONResponse(status_code=503, content={"error": "AI service not configured."})

    # Build rich context message
    parts = [f"Property: {body.address}"]
    if body.features:
        parts.append(f"Key features:\n{body.features}")

    if body.photo_analyses:
        parts.append("\nPhoto analysis:")
        for i, analysis in enumerate(body.photo_analyses):
            desc = analysis.get("description", "")
            room = analysis.get("room_type", "room")
            features = ", ".join(analysis.get("key_features", []))
            parts.append(f"  Photo {i+1} ({room}): {desc}")
            if features:
                parts.append(f"    Notable: {features}")

    if body.area_research:
        research = body.area_research
        parts.append("\nArea research:")
        if research.get("transport"):
            parts.append(f"  Transport: {'; '.join(research['transport'][:5])}")
        if research.get("schools"):
            parts.append(f"  Schools: {'; '.join(research['schools'][:3])}")
        if research.get("amenities"):
            parts.append(f"  Amenities: {'; '.join(research['amenities'][:5])}")
        if research.get("lifestyle"):
            parts.append(f"  Lifestyle: {research['lifestyle']}")

    user_message = "\n".join(parts)

    async def stream():
        start_time = time.time()
        full_text = ""
        current_variant = 1
        variant_text = ""

        try:
            with claude_client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                system=DESCRIPTIONS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as s:
                for text in s.text_stream:
                    full_text += text
                    variant_text += text

                    # Check for variant separator
                    if "---VARIANT---" in variant_text:
                        # Split on marker
                        before_marker = variant_text.split("---VARIANT---")[0]
                        # Emit remaining text before marker
                        if before_marker.strip():
                            yield f"event: token\ndata: {json.dumps({'text': before_marker.rstrip(), 'variant': current_variant})}\n\n"

                        word_count = len(variant_text.split("---VARIANT---")[0].split())
                        yield f"event: variant_complete\ndata: {json.dumps({'variant': current_variant, 'word_count': word_count})}\n\n"

                        current_variant += 1
                        # Keep text after marker
                        variant_text = variant_text.split("---VARIANT---", 1)[1]
                        # Don't emit the variant_text remainder yet — it'll be emitted as tokens
                        if variant_text.strip():
                            yield f"event: token\ndata: {json.dumps({'text': variant_text.lstrip(), 'variant': current_variant})}\n\n"
                            variant_text = ""
                    else:
                        # Normal token — emit if no pending marker check needed
                        # Buffer a bit to avoid splitting the marker
                        if len(variant_text) > 20 and "---" not in variant_text[-20:]:
                            emit_text = variant_text[:-20]
                            variant_text = variant_text[-20:]
                            yield f"event: token\ndata: {json.dumps({'text': emit_text, 'variant': current_variant})}\n\n"

                # Flush remaining text
                if variant_text.strip():
                    yield f"event: token\ndata: {json.dumps({'text': variant_text, 'variant': current_variant})}\n\n"

                # Final variant complete
                yield f"event: variant_complete\ndata: {json.dumps({'variant': current_variant, 'word_count': len(variant_text.split())})}\n\n"

            elapsed_ms = int((time.time() - start_time) * 1000)
            total_words = len(full_text.replace("---VARIANT---", "").split())
            yield f"event: done\ndata: {json.dumps({'variants': current_variant, 'total_words': total_words, 'generation_time_ms': elapsed_ms})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': f'Description generation failed: {str(e)[:100]}'})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ============================================================
# WORKFLOW: Process Photo (Decor8 AI proxy)
# ============================================================


@app.post("/api/workflow/process-photo")
async def process_photo(request: Request, body: ProcessPhotoRequest):
    if not DECOR8_API_KEY:
        return JSONResponse(status_code=503, content={"error": "Image processing service not configured."})

    if not body.image_data_url.startswith("data:image/"):
        return JSONResponse(status_code=400, content={"error": "Invalid image data URL."})

    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=120.0) as http_client:

            if body.processing_type == "stage":
                # Virtual Staging — /generate_designs_for_room
                resp = await http_client.post(
                    f"{DECOR8_BASE_URL}/generate_designs_for_room",
                    headers={
                        "Authorization": f"Bearer {DECOR8_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input_image_url": body.image_data_url,
                        "room_type": body.room_type,
                        "design_style": "modern",
                        "num_images": 1,
                    },
                )

                if resp.status_code != 200:
                    error_text = resp.text[:200]
                    return JSONResponse(status_code=502, content={"error": f"Staging failed: {error_text}"})

                data = resp.json()
                images = data.get("info", {}).get("images", [])
                if not images:
                    return JSONResponse(status_code=502, content={"error": "No staged image returned."})

                processed_url = images[0].get("url", "")

            elif body.processing_type == "enhance":
                # Image Enhancement — /upscale_image (FREE at 2x)
                # This endpoint uses multipart/form-data
                b64_data = body.image_data_url.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_data)

                # Determine file extension from data URL
                mime = body.image_data_url.split(";")[0].split(":")[1]
                ext = "jpg" if "jpeg" in mime else mime.split("/")[1]

                resp = await http_client.post(
                    f"{DECOR8_BASE_URL}/upscale_image",
                    headers={"Authorization": f"Bearer {DECOR8_API_KEY}"},
                    files={"input_image": (f"photo.{ext}", img_bytes, mime)},
                    data={"scale_factor": "2"},
                )

                if resp.status_code != 200:
                    error_text = resp.text[:200]
                    return JSONResponse(status_code=502, content={"error": f"Enhancement failed: {error_text}"})

                data = resp.json()
                upscaled_b64 = data.get("info", {}).get("upscaled_image", "")
                if not upscaled_b64:
                    return JSONResponse(status_code=502, content={"error": "No enhanced image returned."})

                # Return as data URL (upscale returns base64)
                processed_url = f"data:image/jpeg;base64,{upscaled_b64}"

            elif body.processing_type == "declutter":
                # Furniture Removal — /remove_objects_from_room
                resp = await http_client.post(
                    f"{DECOR8_BASE_URL}/remove_objects_from_room",
                    headers={
                        "Authorization": f"Bearer {DECOR8_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"input_image_url": body.image_data_url},
                )

                if resp.status_code != 200:
                    error_text = resp.text[:200]
                    return JSONResponse(status_code=502, content={"error": f"Declutter failed: {error_text}"})

                data = resp.json()
                image_info = data.get("info", {}).get("image", {})
                processed_url = image_info.get("url", "")
                if not processed_url:
                    return JSONResponse(status_code=502, content={"error": "No decluttered image returned."})

            else:
                return JSONResponse(status_code=400, content={"error": f"Unknown processing type: {body.processing_type}"})

        elapsed_ms = int((time.time() - start_time) * 1000)

        return JSONResponse(content={
            "success": True,
            "processed_image_url": processed_url,
            "processing_type": body.processing_type,
            "generation_time_ms": elapsed_ms,
        })

    except httpx.TimeoutException:
        return JSONResponse(status_code=504, content={"error": "Image processing timed out. Please try again."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)[:150]}"})


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
