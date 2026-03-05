# Wallet Manufacturing BOM AI 👛
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14osf7-Pu7TtgEu9-BU3M_RwwoPfdJb8P?usp=sharing)

This repository contains an end-to-end pipeline to automate the creation of **Manufacturing Bill of Materials (BOM)** from wallet product images. It uses model distillation (Gemini to BLIP) to identify technical construction details like stitching, edge finishing, and hardware components.

## 🎯 The Core Mission
Transforming visual product designs into actionable engineering data. Instead of just describing a wallet, this AI generates a structured "recipe" that a factory can use to initiate the manufacturing process.

### **BOM Output Example:**
| COMPONENT | MANUFACTURING SPECIFICATION |
| :--- | :--- |
| **Material (Primary)** | Textured Brown Leather |
| **Product Architecture** | Compact Cardholder |
| **Closure Mechanism** | Fold Closure |
| **Assembly Process** | Machine-Stitched |
| **Edge Specification** | Folded Edges |

---

## 📂 Repository Structure
*   `wallet/` - Dataset of 476 wallet product images.
*   `wallet_captions.json` - Distilled 12-field manufacturing ground-truth (BOM).
*   `generate_captions.py` - Script using Gemini 3 Flash to extract BOM data from raw images.
*   `colab_trainer.ipynb` - The training notebook for fine-tuning the BLIP student model.

---

## 🚀 Phase 1: BOM Dataset Generation
We use **Gemini 3 Flash** as a high-fidelity "Teacher" to audit wallet images for technical features that are often missed by standard captioning models.

**Extracted BOM Fields (10 Construction-Only Fields):**
1. Wallet Type (Bifold, Trifold, etc.)
2. Color
3. Surface Pattern / Texture
4. Primary Material (Leather, Canvas, etc.)
5. Hardware Components (Zippers, Snaps, Badges)
6. Closure Type
7. Stitching Type
8. Stitch Color
9. Edge Finishing (Painted, Burnished, Folded)
10. Size Category

---

## 🧠 Phase 2: Student Model Training 
We fine-tune `Salesforce/blip-image-captioning-base` on the distilled BOM data.

**Training Stats (Tuned):**
*   **Dataset:** 476 Images / 10-field BOM Captions (no brand data).
*   **Optimizer:** AdamW (LR: 2e-5, Weight Decay: 0.01).
*   **Batch Size:** 8.
*   **Training:** 3 Epochs (tuned for better generalization — lower LR + weight decay prevents memorization).

---

## 💾 Using the Trained Model
1.  **Download weights:** [Google Drive - finetuned_bom_model_final.zip](https://drive.google.com/drive/folders/1EZJRpYl-GWtAaqT_UldQHeiv46XulIAT?usp=sharing)
2.  **Run locally:**
```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("./finetuned_wallet_blip")
model = BlipForConditionalGeneration.from_pretrained("./finetuned_wallet_blip")

image = Image.open("sample.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(out[0], skip_special_tokens=True))
```