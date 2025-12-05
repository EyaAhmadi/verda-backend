from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from verda import GenerativeVerdaAgent
import shutil
import os

# --- Class names for model ---
CLASS_NAMES = [
    'Apple___apple_scab','Apple___black_rot','Apple___cedar_apple_rust','Apple___healthy',
    'Bell_pepper___bacterial_spot','Bell_pepper___healthy','Cherry___healthy','Cherry___powdery_mildew',
    'Corn_maize___cercospora_leaf_spot','Corn_maize___common_rust','Corn_maize___healthy','Corn_maize___northern_leaf_blight',
    'Grape___black_rot','Grape___esca_(black_measles)','Grape___healthy','Grape___leaf_blight',
    'Peach___bacterial_spot','Peach___healthy','Potato___early_blight','Potato___healthy','Potato___late_blight',
    'Strawberry___healthy','Strawberry___leaf_scorch ',
    'Tomato___bacterial_spot','Tomato___early_blight','Tomato___healthy','Tomato___late_blight',
    'Tomato___leaf_mold','Tomato___septoria_leaf_spot','Tomato___yellow_leaf_curl_virus'
]

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load GenerativeVerdaAgent
plant_agent = GenerativeVerdaAgent("checkpoints/fine_tuned_mobilenet.pth", CLASS_NAMES)


@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    # Validate file size (max 10MB for eco-efficiency)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={"error": "Image trop volumineuse. Max 10MB."}
        )
    
    os.makedirs("temp", exist_ok=True)
    path = f"temp/{file.filename}"

    # Save upload
    with open(path, "wb") as f:
        f.write(contents)

    try:
        # Run model with LLM recommendation generation
        result = plant_agent.identify_plant(path, lang="fr")

        # Return compact response
        return {
            "plant_name": result["plant_name"],
            "disease_status": result["disease_status"],
            "confidence": round(result["confidence"], 2),
            "recommendation": result["recommendation"]
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur d'analyse: {str(e)[:100]}"}
        )
    
    finally:
        # Eco cleanup
        try:
            os.remove(path)
        except:
            pass


@app.get("/health")
async def health_check():
    """Lightweight health check endpoint"""
    return {"status": "ok"}