from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io

app = FastAPI(title="BeautyTech AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
CLASS_NAMES = ['Acne', 'Normal']

# Load model
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model.load_state_dict(torch.load('./skin_acne_model.pth', map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Ngưỡng tự tin tối thiểu
CONFIDENCE_THRESHOLD = 0.7


@app.get("/")
def root():
    return {"message": "BeautyTech AI Service is running", "device": str(device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            _, pred = torch.max(outputs, 1)

        predicted_class = CLASS_NAMES[pred.item()]
        confidence = probabilities[pred.item()].item()

        probs_dict = {
            CLASS_NAMES[i]: round(probabilities[i].item(), 4)
            for i in range(NUM_CLASSES)
        }

        # Nếu model không đủ tự tin → trả về "Uncertain"
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "success": True,
                "skin_condition": "Uncertain",
                "confidence": round(confidence, 4),
                "probabilities": probs_dict,
                "message": "Không đủ tự tin. Vui lòng chụp ảnh rõ hơn, đủ sáng, mặt thẳng."
            }

        return {
            "success": True,
            "skin_condition": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": probs_dict
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)