import os
import json
import time
import concurrent.futures
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
import threading

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemma-3-27b-it"

# ───────────────────────────────────────────────────────────────
# Expanded Pydantic Schema: Bill of Materials for Manufacturing
# ───────────────────────────────────────────────────────────────
class WalletBOM(BaseModel):
    """Bill of Materials schema for a wallet image — designed for manufacturing.
    Only includes features reliably visible from exterior product photos."""

    # Core Identity
    wallet_type: str = Field(description="Type of wallet: bifold, trifold, zip-around, cardholder, clutch, money-clip, passport-holder, wristlet, phone-wallet, etc.")
    color: str = Field(description="Primary color(s) of the wallet, e.g. 'brown', 'black and tan', 'burgundy'")
    pattern: str = Field(description="Surface pattern: solid, textured, embossed-crocodile, monogram-print, woven, quilted, perforated, pebbled, plain, etc.")

    # Exterior Material
    primary_material: str = Field(description="Main body material: full-grain leather, top-grain leather, PU/synthetic leather, canvas, nylon, polyester, cork, vegan leather, fabric, etc.")

    # Hardware (visible from exterior)
    hardware_components: List[str] = Field(description="List of all hardware visible on the exterior: e.g. ['zipper', 'snap button', 'D-ring', 'rivets', 'magnetic clasp', 'metal logo badge', 'chain', 'buckle']. Use empty list if none visible.")
    closure_type: str = Field(description="How the wallet closes: fold, zipper, snap-button, magnetic-snap, velcro, open, button-strap, flap, etc.")

    # Stitching & Construction (visible from exterior)
    stitching_type: str = Field(description="Type of stitching visible on exterior: machine-stitch, saddle-stitch, hand-stitch, edge-stitch, contrast-stitch, or not-visible")
    stitch_color: str = Field(description="Color of the stitching thread visible on exterior, e.g. 'beige', 'matching', 'contrast white', or 'not-visible'")
    edge_finish: str = Field(description="Edge finishing visible: painted, burnished, raw, folded, bound, or not-visible")

    # Branding (visible from exterior)
    brand: str = Field(description="Brand name if identifiable from logo/text on exterior, otherwise 'unbranded'")
    branding_method: str = Field(description="How the brand is applied on exterior: embossed, debossed, heat-stamp, printed, metal-badge, stitched-label, engraved, none")

    # Size Estimate
    size_category: str = Field(description="Estimated size category based on proportions: compact, standard, large, oversized")


progress_lock = threading.Lock()
completed_count = 0
total_images = 0

def process_image(img_path):
    global completed_count, total_images
    
    attempt = 0
    while True:
        try:
            img = Image.open(img_path)
            prompt = (
                "You are a manufacturing engineer analyzing a wallet product image. "
                "Your job is to extract a complete Bill of Materials (BOM) from the image. "
                "Look carefully at the wallet and identify:\n"
                "- The type of wallet and its core materials (exterior and lining)\n"
                "- All design components: card slots, bill compartments, coin pockets, ID windows\n"
                "- All hardware: zippers, snaps, rivets, clasps, D-rings, metal badges\n"
                "- Closure mechanism and stitching details (type, color, edge finish)\n"
                "- Branding method and brand name if visible\n"
                "- Color, pattern, and estimated size\n\n"
                "Be precise and specific. If something is not visible, use 'not-visible' or reasonable defaults. "
                "Return the structured JSON."
            )
            
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, img],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=WalletBOM,
                    temperature=0.1,
                )
            )
            
            if response.parsed:
                if hasattr(response.parsed, 'model_dump'): data = response.parsed.model_dump()
                else: data = response.parsed.dict()
            else:
                result_str = response.text.strip()
                if result_str.startswith("```json"): result_str = result_str[7:]
                if result_str.startswith("```"): result_str = result_str[3:]
                if result_str.endswith("```"): result_str = result_str[:-3]
                data = json.loads(result_str.strip())
            
            with progress_lock:
                completed_count += 1
                progress_percentage = (completed_count / max(total_images, 1)) * 100
                print(f"[{completed_count}/{total_images} | {progress_percentage:.1f}%] ✅ Successfully processed: {img_path.name}")
                
            return str(img_path), data
            
        except Exception as e:
            attempt += 1
            if "503" in str(e) or "429" in str(e) or "quota" in str(e).lower():
                wait_time = min(30 * attempt, 180)
                print(f"⚠️ Congestion (503/429) on {img_path.name}. Attempt {attempt}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                wait_time = 10
                print(f"⚠️ Error processing {img_path.name}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

def main():
    global total_images, completed_count
    folder_path = Path("wallet")
    
    # Output file — fresh start with BOM schema
    output_file = Path("wallet_captions.json")
    results = {}
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing = json.loads(content)
                    # Only reuse entries that have the new BOM fields
                    for key, val in existing.items():
                        if "primary_material" in val and "card_slots" in val:
                            results[key] = val
                    if results:
                        print(f"Loaded {len(results)} existing BOM results from {output_file}.")
                    else:
                        print(f"Old format detected in {output_file}. Starting fresh with new BOM schema.")
        except Exception as e:
            print(f"Could not load existing file: {e}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    all_image_paths = [p for p in folder_path.iterdir() if p.suffix.lower() in valid_extensions]
    
    # Only process what hasn't been done yet
    image_paths_to_process = [p for p in all_image_paths if str(p) not in results]
    
    total_images = len(image_paths_to_process)
    completed_count = 0
    
    print(f"\n📊 Total images in folder: {len(all_image_paths)}")
    print(f"📊 Already processed (BOM): {len(results)}")
    print(f"📊 Remaining to process: {total_images}")
    
    if total_images == 0:
        print("All images are already processed! Dataset is complete.")
        return
        
    max_workers = 8 
    print(f"\n🚀 Executing with {max_workers} parallel workers. Incremental saves enabled...\n")
    
    file_save_lock = threading.Lock()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {executor.submit(process_image, img_path): img_path for img_path in image_paths_to_process}
        
        for future in concurrent.futures.as_completed(future_to_image):
            img_path = future_to_image[future]
            try:
                path_str, data = future.result()
                
                with file_save_lock:
                    results[path_str] = data
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=4)
                    
            except Exception as exc:
                print(f"Critical Exception on {img_path.name} from Executor: {exc}")
                
    print(f"\n🎉 DONE! 100% complete. {len(results)} wallet BOMs saved in '{output_file}'.")

if __name__ == "__main__":
    main()
