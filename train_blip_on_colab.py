# Google Colab Training Script for Finetuning BLIP on Wallet Captions
# 
# INSTRUCTIONS FOR GOOGLE COLAB:
# 1. Open Google Colab (colab.research.google.com) and create a New Notebook.
# 2. Go to Runtime > Change runtime type > Select "T4 GPU" (or better).
# 3. Zip your local "wallet" folder and "wallet_captions.json" -> upload them to Colab and unzip.
# 4. In a Colab cell, install dependencies: 
#    !pip install -q transformers torch torchvision Pillow
# 5. Copy and run this script in the next cell!

import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.optim import AdamW
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "./"  # Directory containing 'wallet' folder and 'wallet_captions.json'
JSON_FILE = os.path.join(DATA_DIR, "wallet_captions.json")
MODEL_ID = "Salesforce/blip-image-captioning-base"
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
SAVE_PATH = "./finetuned_wallet_blip"

# --- DATA PREPARATION ---
class WalletDataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data_dict = json.load(f)
            
        self.image_paths = list(self.data_dict.keys())
        self.processor = processor
        
    def __len__(self):
        return len(self.image_paths)
    
    def _create_caption(self, features):
        """Converts the JSON structure into a natural language sentence for BLIP to learn."""
        color = features.get("color", "unknown").lower()
        material = features.get("material_type", "unknown").lower()
        wallet_type = features.get("type_of_wallet", "wallet").lower()
        pattern = features.get("pattern", "solid").lower()
        brand = features.get("brand", "unknown").lower()
        
        # Format a descriptive text sequence
        caption = f"a {pattern} {color} {material} {wallet_type}"
        if brand != "unknown" and brand != "":
            caption += f" by {brand}"
            
        # Example output: "a solid black leather bifold by fossil"
        return caption

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        features = self.data_dict[img_path]
        
        # Format the actual path for Colab (handles Windows \\ paths from original generation)
        # Assuming we uploaded the folder "wallet" to the same colab directory
        # We strip the original "wallet\\" prefix if it exists to make it work on Linux Colab
        pure_filename = os.path.basename(img_path)
        colab_img_path = os.path.join(DATA_DIR, "wallet", pure_filename)
        
        try:
            image = Image.open(colab_img_path).convert("RGB")
        except Exception as e:
            # Fallback for missing images
            print(f"Skipping {colab_img_path}: {e}")
            image = Image.new("RGB", (224, 224), (255, 255, 255)) 

        caption = self._create_caption(features)
        
        # Process the image and text to turn them into model tensor inputs
        encoding = self.processor(images=image, text=caption, padding="max_length", return_tensors="pt")
        
        # Remove batch dimension added by the processor since DataLoader groups them
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding

def main():
    print(f"Loading processor and model: {MODEL_ID}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Load processor and Base Model
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    model.to(device)

    # Prepare Dataset & DataLoader
    dataset = WalletDataset(JSON_FILE, processor)
    print(f"Loaded dataset with {len(dataset)} examples.")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Starting Training Loop...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            # Move inputs to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            
            # The model requires labels for calculating the loss during training
            # In transformers, input_ids are used as labels for causal LM tasks
            labels = input_ids.clone()
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass & optimize
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            loop.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete - Average Loss: {avg_loss:.4f}")

    print("Training finished! Saving the fine-tuned model...")
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    processor.save_pretrained(SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}. You can now use it for inference!")

    # --- INFERENCE EXAMPLE ---
    print("\n[Sanity Check] Running test inference on an image...")
    model.eval()
    test_img_path = os.path.join(DATA_DIR, "wallet", os.path.basename(dataset.image_paths[0]))
    try:
        test_img = Image.open(test_img_path).convert("RGB")
        inputs = processor(test_img, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        predicted_caption = processor.decode(out[0], skip_special_tokens=True)
        print(f"Original JSON -> {dataset.data_dict[dataset.image_paths[0]]}")
        print(f"Model Prediction -> {predicted_caption}")
    except Exception as e:
        print("Couldn't run inference check:", e)

if __name__ == "__main__":
    main()
