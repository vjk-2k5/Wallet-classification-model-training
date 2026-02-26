# Wallet Classification & Description Model Training 👛
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14osf7-Pu7TtgEu9-BU3M_RwwoPfdJb8P?usp=sharing)

This repository contains the complete pipeline for generating a highly structured, custom image captioning dataset and fine-tuning a Vision-Language Model (Salesforce BLIP) to accurately describe wallet images.

## Project Overview

The goal of this project is to take raw images of wallets and train an AI to output precise, structured descriptions in a very specific format:
> `a [pattern] [color] [material_type] [type_of_wallet] by [brand]`

*(e.g., "a solid brown leather bifold wallet by fossil" or "a solid black synthetic zip-around by unbranded")*

To achieve this, the pipeline is divided into two parts:
1. **Automated Dataset Generation:** Using Google's Gemini 3 Flash API to analyze 250+ images and automatically extract structured JSON data about each wallet's material, color, type, brand, and pattern.
2. **Model Fine-Tuning:** Training a lightweight `Salesforce/blip-image-captioning-base` model on Google Colab to learn this exact descriptive format.

---

## 📂 Repository Structure

*   `wallet/` - Directory containing the raw images of the wallets (not included in the repo by default).
*   `wallet_captions.json` - The generated dataset mapping image paths to their structured attributes.
*   `dataset.zip` - A compressed archive of the images and JSON, ready for Colab upload.
*   `generate_captions.py` - Script that uses the Gemini GenAI API with parallel processing to build the dataset.
*   `train_blip_on_colab.py` - A standalone Python script containing the PyTorch dataset loaders and training loop.
*   `colab_trainer.ipynb` - The primary Google Colab Notebook used to actually run the fine-tuning on a T4 GPU.

---

## 🚀 Phase 1: Generating the Dataset (`generate_captions.py`)

Takes local images and generates a highly accurate JSON ground-truth dataset.

**Features:**
*   Uses `gemini-3-flash-preview` to look at images and return rigid JSON using Pydantic schemas.
*   Multithreaded with dynamic API Rate-Limit handling (infinite retry with exponential backoff on 429/503 errors).
*   Incremental appending saves state so no data is lost if the process fails.

**How to run:**
1. Put your images in the `wallet/` directory.
2. Create a `.env` file and add your Google API key: `GEMINI_API_KEY=your_key_here`
3. Run the script:
   ```bash
   python generate_captions.py
   ```
4. This outputs `wallet_captions.json` and a zipped version `dataset.zip`.

---

## 🧠 Phase 2: Fine-Tuning BLIP (`colab_trainer.ipynb`)

We use Google Colab to take advantage of free T4 GPUs to radically speed up the training of the BLIP Base model. 

**Features:**
*   Automatically resolves Windows-to-Linux pathing issues.
*   Safely formats the JSON attributes into a natural language sentence.
*   Loss tracking and automatic visual graphing.
*   Built-in "Anti-Cheat" unseen image fetching to test for overfitting at the end.

**How to run:**
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `colab_trainer.ipynb`.
3. Change Runtime to **T4 GPU** (`Runtime > Change runtime type`).
4. Upload `dataset.zip` into the Colab file explorer on the left.
5. Click **Run All**.

*(Note: Optimal training time for ~250 images is **~3 Epochs** to avoid overfitting. The notebook takes about 5 minutes to run.)*

---

## 💾 Using the Trained Model
**📥 Download the pre-trained custom model weights here:** 
[Google Drive Link - finetuned_wallet_blip](https://drive.google.com/drive/folders/1EZJRpYl-GWtAaqT_UldQHeiv46XulIAT?usp=sharing)

After downloading, extract the folder so it is located at `./finetuned_wallet_blip` relative to your script.

You can now use your custom AI locally with HuggingFace `transformers`:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("./finetuned_wallet_blip")
model = BlipForConditionalGeneration.from_pretrained("./finetuned_wallet_blip")

image = Image.open("any_new_wallet.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(out[0], skip_special_tokens=True))
# Output: "a solid red leather trifold by unbranded"
```