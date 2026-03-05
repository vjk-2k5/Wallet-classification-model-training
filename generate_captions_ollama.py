"""
generate_captions_ollama.py — Open-Source BOM Caption Generator using Ollama

============================================================================
WHY THIS FILE EXISTS:
Google's Terms of Service prohibit using Gemini API outputs to train or
distill competing AI models for commercial use. If you plan to use the
fine-tuned BLIP model commercially, you MUST use an open-source teacher
model instead.

This script uses Ollama (https://ollama.com) to run open-source vision
models locally (e.g., LLaVA, Gemma 3) for generating the same BOM
captions — fully license-compliant for commercial distillation.
============================================================================

Usage:
  1. Install Ollama: https://ollama.com/download
  2. Pull a vision model:  ollama pull llava:13b   (or gemma3:12b)
  3. Create a .env file:   (no API key needed!)
  4. Run:  python generate_captions_ollama.py
"""

import os
import json
import time
import concurrent.futures
from pathlib import Path
from PIL import Image
import requests
import threading
import base64
import io

# ───────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llava:13b"  # Change to "gemma3:12b" or any vision model you have

# ───────────────────────────────────────────────────────────────
# BOM Schema (same 10 fields as Gemini version)
# ───────────────────────────────────────────────────────────────
BOM_SCHEMA = {
    "wallet_type": "Type of wallet: bifold, trifold, zip-around, cardholder, clutch, money-clip, etc.",
    "color": "Primary color(s), e.g. 'brown', 'black and tan'",
    "pattern": "Surface pattern: solid, textured, embossed-crocodile, quilted, pebbled, plain, etc.",
    "primary_material": "Main body material: full-grain leather, PU/synthetic leather, canvas, nylon, etc.",
    "hardware_components": "List of visible hardware: zipper, snap button, D-ring, rivets, buckle. Empty list if none.",
    "closure_type": "How it closes: fold, zipper, snap-button, magnetic-snap, velcro, open, flap, etc.",
    "stitching_type": "Stitching visible: machine-stitch, saddle-stitch, hand-stitch, edge-stitch, or not-visible",
    "stitch_color": "Thread color: beige, matching, contrast white, or not-visible",
    "edge_finish": "Edge finishing: painted, burnished, raw, folded, bound, or not-visible",
    "size_category": "Size estimate: compact, standard, large, oversized",
}

PROMPT = """You are a manufacturing engineer analyzing a wallet product image.
Your job is to extract a Bill of Materials (BOM) for manufacturing — focus ONLY on construction details.

Look carefully at the wallet and identify:
- The type of wallet (bifold, trifold, zip-around, cardholder, etc.)
- Exterior material (leather type, synthetic, canvas, etc.)
- Color and surface pattern/texture
- All hardware: zippers, snaps, rivets, clasps, D-rings, buckles
- Closure mechanism
- Stitching details (type, thread color, edge finish)
- Estimated size category

Do NOT identify brand names. Focus purely on materials and construction processes.

Return ONLY a valid JSON object with these exact keys:
{
    "wallet_type": "...",
    "color": "...",
    "pattern": "...",
    "primary_material": "...",
    "hardware_components": ["...", "..."],
    "closure_type": "...",
    "stitching_type": "...",
    "stitch_color": "...",
    "edge_finish": "...",
    "size_category": "..."
}

Return ONLY the JSON object, no explanations or markdown."""

# ───────────────────────────────────────────────────────────────
# Globals
# ───────────────────────────────────────────────────────────────
progress_lock = threading.Lock()
completed_count = 0
total_images = 0


def image_to_base64(img_path):
    """Convert image to base64 string for Ollama API."""
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        # Resize if too large (Ollama can struggle with huge images)
        max_dim = 1024
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def process_image(img_path):
    """Process a single image through Ollama vision model."""
    global completed_count, total_images
    
    attempt = 0
    while True:
        try:
            img_b64 = image_to_base64(img_path)
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": PROMPT,
                    "images": [img_b64],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 512,
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            
            result_text = response.json()["response"].strip()
            
            # Clean up response — extract JSON from possible markdown blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            # Find the JSON object
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start >= 0 and end > start:
                result_text = result_text[start:end]
            
            data = json.loads(result_text)
            
            # Validate all required keys exist
            for key in BOM_SCHEMA:
                if key not in data:
                    data[key] = "not-visible" if key != "hardware_components" else []
            
            with progress_lock:
                completed_count += 1
                pct = (completed_count / max(total_images, 1)) * 100
                print(f"[{completed_count}/{total_images} | {pct:.1f}%] ✅ {img_path.name}")
                
            return str(img_path), data
            
        except json.JSONDecodeError as e:
            attempt += 1
            print(f"⚠️ JSON parse error on {img_path.name} (attempt {attempt}): {e}")
            if attempt >= 5:
                print(f"❌ Skipping {img_path.name} after {attempt} failed attempts.")
                return str(img_path), None
            time.sleep(2)
            
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Ollama at {OLLAMA_BASE_URL}. Is Ollama running?")
            print(f"   Start it with: ollama serve")
            time.sleep(5)
            attempt += 1
            if attempt >= 3:
                raise ConnectionError(f"Ollama is not running at {OLLAMA_BASE_URL}")
                
        except Exception as e:
            attempt += 1
            wait_time = min(10 * attempt, 60)
            print(f"⚠️ Error on {img_path.name}: {e}. Retry in {wait_time}s...")
            time.sleep(wait_time)
            if attempt >= 5:
                print(f"❌ Skipping {img_path.name} after {attempt} attempts.")
                return str(img_path), None


def main():
    global total_images, completed_count
    folder_path = Path("wallet")
    
    output_file = Path("wallet_captions.json")
    results = {}
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing = json.loads(content)
                    for key, val in existing.items():
                        if "primary_material" in val and "stitching_type" in val and "brand" not in val:
                            results[key] = val
                    if results:
                        print(f"Loaded {len(results)} existing BOM results from {output_file}.")
                    else:
                        print(f"Old format detected. Starting fresh with new BOM schema.")
        except Exception as e:
            print(f"Could not load existing file: {e}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    all_image_paths = [p for p in folder_path.iterdir() if p.suffix.lower() in valid_extensions]
    
    image_paths_to_process = [p for p in all_image_paths if str(p) not in results]
    
    total_images = len(image_paths_to_process)
    completed_count = 0
    
    print(f"\n📊 Total images in folder: {len(all_image_paths)}")
    print(f"📊 Already processed (BOM): {len(results)}")
    print(f"📊 Remaining to process: {total_images}")
    print(f"🤖 Using Ollama model: {MODEL_NAME}")
    
    if total_images == 0:
        print("All images are already processed! Dataset is complete.")
        return
    
    # Ollama runs locally — use fewer workers to avoid overwhelming your machine
    # Increase if you have a powerful GPU (e.g., RTX 3090/4090)
    max_workers = 2
    print(f"\n🚀 Processing with {max_workers} parallel workers (local GPU)...\n")
    
    file_save_lock = threading.Lock()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {executor.submit(process_image, img_path): img_path for img_path in image_paths_to_process}
        
        for future in concurrent.futures.as_completed(future_to_image):
            img_path = future_to_image[future]
            try:
                path_str, data = future.result()
                
                if data is not None:
                    with file_save_lock:
                        results[path_str] = data
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=4)
                    
            except Exception as exc:
                print(f"Critical Exception on {img_path.name}: {exc}")
                
    print(f"\n🎉 DONE! {len(results)} wallet BOMs saved in '{output_file}'.")


if __name__ == "__main__":
    main()
