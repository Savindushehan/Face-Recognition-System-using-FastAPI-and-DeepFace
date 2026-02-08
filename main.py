from  fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepface import DeepFace
import shutil
import os

# ---------------- CONFIG ----------------
DB_PATH = "face_db"            # folder with student images
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "mtcnn"     # or "retinaface"
DISTANCE_THRESHOLD = 0.65
UPLOAD_FOLDER = "uploads"      # temporary uploaded files
# ---------------------------------------

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI(title="Face Recognition API")

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """
    Upload an image and get the recognized student's identity.
    """
    # Save uploaded file temporarily
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    label = "Unknown"

    try:
        # DeepFace recognition
        results = DeepFace.find(
            img_path=file_path,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            silent=True
        )

        # Check best match
        if len(results) > 0 and len(results[0]) > 0:
            best_match = results[0].iloc[0]
            distance = best_match["distance"]
            if distance < DISTANCE_THRESHOLD:
                identity_path = best_match["identity"]
                label = os.path.basename(os.path.dirname(identity_path))

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

    return JSONResponse(content={"identity": label})
