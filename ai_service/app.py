from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import os
import gdown
from torchvision import transforms
from transformers import AutoTokenizer
from test2 import MultimodalClassifier

app = FastAPI(title="Fake News Detector Engine")

# Enable CORS so your Node.js backend can talk to this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AUTO-DOWNLOAD MODEL WEIGHTS FROM GOOGLE DRIVE ---
# Dynamically resolves path so it works perfectly in cloud environment containers
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_multimodal_model (1).pth")

# ⚠️ PLACE YOUR GOOGLE DRIVE FILE ID IN THE STRING BELOW
GOOGLE_DRIVE_LINK = "https://drive.google.com/file/d/1htEDPHuBfNQmyjexrgXOHmY02uWiADoL/view?usp=sharing"

if not os.path.exists(MODEL_PATH):
    print("🤖 Model weights not found locally. Downloading from Google Drive...")
    try:
        gdown.download(GOOGLE_DRIVE_LINK, MODEL_PATH, quiet=False)
        print("✅ Model weights downloaded successfully!")
    except Exception as e:
        print(f"❌ Failed to download model weights: {e}")
# -----------------------------------------------------

# Load the model weights onto the available hardware (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(headline_text: str = Form(""), image_file: UploadFile = File(None)):
    has_text = bool(headline_text.strip())
    
    # Process text input channel
    if has_text:
        text_inputs = tokenizer(headline_text, max_length=128, padding='max_length', truncation=True, return_tensors="pt").to(device)
    else:
        text_inputs = tokenizer("", max_length=128, padding='max_length', truncation=True, return_tensors="pt").to(device)

    # Process image input channel safely
    if image_file is not None:
        try:
            image_bytes = await image_file.read()
            raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = img_transform(raw_image).unsqueeze(0).to(device)
        except Exception:
            image_tensor = torch.zeros(1, 3, 224, 224).to(device)
    else:
        image_tensor = torch.zeros(1, 3, 224, 224).to(device)

    # Execute model prediction pass
    with torch.no_grad():
        outputs = model(text_inputs['input_ids'], text_inputs['attention_mask'], image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    label_verdict = "Fake" if prediction.item() == 1 else "Real"
    
    return {
        "verdict": label_verdict,
        "confidence": round(confidence.item() * 100, 2),
        "text_analyzed": has_text,
        "image_analyzed": image_file is not None
    }

if __name__ == "__main__":
    import uvicorn
    # Changed host to 0.0.0.0 for external exposure in deployment environments
    uvicorn.run(app, host="0.0.0.0", port=8000)