import os
import json
import time
import concurrent.futures
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
import threading

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3-flash-preview"

class WalletInfo(BaseModel):
    material_type: str
    color: str
    type_of_wallet: str
    brand: str
    pattern: str

progress_lock = threading.Lock()
completed_count = 0
total_images = 0

def process_image(img_path):
    global completed_count, total_images
    
    attempt = 0
    # Infinite loop to keep retrying with exponential backoff until it succeeds
    while True:
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
                wait_time = min(30 * attempt, 180) # Back-off up to 3 minutes
                print(f"⚠️ Congestion (503/429) on {img_path.name}. Attempt {attempt}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                wait_time = 10
                print(f"⚠️ Error processing {img_path.name}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

def main():
    global total_images, completed_count
    folder_path = Path("wallet")
    
    # Load previously saved progress if any
    output_file = Path("wallet_captions.json")
    results = {}
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    results = json.loads(content)
                    print(f"Loaded {len(results)} existing results from {output_file}.")
        except Exception as e:
            print(f"Could not load existing file: {e}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    all_image_paths = [p for p in folder_path.iterdir() if p.suffix.lower() in valid_extensions]
    
    # Only process exactly what hasn't been done yet
    image_paths_to_process = [p for p in all_image_paths if str(p) not in results]
    
    total_images = len(image_paths_to_process)
    completed_count = 0
    
    print(f"\nRemaining images to process: {total_images} / {len(all_image_paths)}")
    
    if total_images == 0:
        print("All images are already processed! Dataset is complete.")
        return
        
    # We'll use 8 workers to blast through the rest fast but reasonably
    max_workers = 8 
    print(f"Executing with {max_workers} parallel workers to parse the rest quickly. Appending saves incrementally...")
    
    # A lock specifically so the incremental saving to disk isn't corrupted by race conditions
    file_save_lock = threading.Lock()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {executor.submit(process_image, img_path): img_path for img_path in image_paths_to_process}
        
        for future in concurrent.futures.as_completed(future_to_image):
            img_path = future_to_image[future]
            try:
                path_str, data = future.result()
                
                with file_save_lock:
                    results[path_str] = data
                    # INCREMENTAL SAVE APPEND!
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=4)
                    
            except Exception as exc:
                print(f"Critical Exception on {img_path.name} from Executor: {exc}")
                
    print(f"\nFINALLY DONE! 100% complete. Captions saved in '{output_file}'.")

if __name__ == "__main__":
    main()
