import os
import json
import time
import concurrent.futures
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

# Use the Official 2026 Google GenAI SDK
from google import genai
from google.genai import types
from pydantic import BaseModel
import threading

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit(1)

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3-flash-preview"

class WalletInfo(BaseModel):
    material_type: str
    color: str
    type_of_wallet: str
    brand: str
    pattern: str

# Use a lock to safely increment the progress counter across multiple threads
progress_lock = threading.Lock()
completed_count = 0
total_images = 0

def process_image(img_path, max_retries=3):
    global completed_count, total_images
    
    for attempt in range(max_retries):
        try:
            img = Image.open(img_path)
            prompt = "Analyze this image of a wallet. Identify and return its material type, color, type of wallet, brand, and pattern."
            
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, img],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=WalletInfo,
                    temperature=0.1,
                )
            )
            
            if response.parsed:
                if hasattr(response.parsed, 'model_dump'):
                    data = response.parsed.model_dump() 
                else: 
                    data = response.parsed.dict() 
            else:
                result_str = response.text.strip()
                if result_str.startswith("```json"): result_str = result_str[7:]
                if result_str.startswith("```"): result_str = result_str[3:]
                if result_str.endswith("```"): result_str = result_str[:-3]
                data = json.loads(result_str.strip())
            
            with progress_lock:
                completed_count += 1
                progress_percentage = (completed_count / total_images) * 100
                print(f"[{completed_count}/{total_images} | {progress_percentage:.1f}%] ✅ Successfully processed: {img_path.name}")
                
            return str(img_path), data
            
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                # Handling Rate limits aggressively
                # 429 means too many requests. We need to back off significantly.
                wait_time = 15 * (attempt + 1)
                print(f"[{completed_count}/{total_images}] ⚠️ Rate limited on {img_path.name}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                print(f"[{completed_count}/{total_images}] ⚠️ Error processing {img_path.name} (Attempt {attempt+1}/{max_retries}): {e}. Retrying in 5s...")
                time.sleep(5)
            else:
                print(f"[{completed_count}/{total_images}] ❌ Final Error processing {img_path.name}: {e}")
                return str(img_path), {"error": str(e)}

def main():
    global total_images
    folder_path = Path("wallet")
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = [p for p in folder_path.iterdir() if p.suffix.lower() in valid_extensions]
    total_images = len(image_paths)
    
    print(f"Found {total_images} images in the '{folder_path}' folder.")
    
    results = {}
    
    # We will lower the workers slightly to 10 to avoid getting hit by 429 too aggressively
    # The Gemini API free tier typically handles 15 RPM. Bursting 20 immediately causes rate limits.
    max_workers = 5
    
    print(f"Starting processing with {max_workers} parallel workers to respect rate limits safely...")
    print("-" * 50)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {executor.submit(process_image, img_path): img_path for img_path in image_paths}
        
        for future in concurrent.futures.as_completed(future_to_image):
            img_path = future_to_image[future]
            try:
                path_str, data = future.result()
                results[path_str] = data
            except Exception as exc:
                print(f"Exception on {img_path.name}: {exc}")
                results[str(img_path)] = {"error": str(exc)}
                
    output_file = "wallet_captions.json"
    print("-" * 50)
    print(f"Saving {len(results)} outputs to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nProcessing complete! Captions mapped directly to image paths are saved in '{output_file}'.")

if __name__ == "__main__":
    main()
