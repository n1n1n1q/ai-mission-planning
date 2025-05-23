from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
from datetime import datetime
from pathlib import Path
import uuid
import random

UPLOAD_DIR = Path("./back/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

with open("back/.env_info", "r", encoding="utf-8") as f:
    uri = f.read().strip()
    

client = MongoClient(uri, server_api=ServerApi('1'))
db = client["etdh"]
videos_collection = db["videos"]

app = FastAPI()

app.mount("/static/videos", StaticFiles(directory=UPLOAD_DIR), name="videos")

class TimestampsModel(BaseModel):
    event: str = None
    start: str = None
    end: str = None

class VideoCreate(BaseModel):
    video_name: str
    file_path: str
    timestamps: list[TimestampsModel] = None

@app.post("/api/videos/", status_code=201)
async def create_video(
    video_name: str,
    video_file: UploadFile 
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    file_extension = os.path.splitext(video_file.filename)[1]
    safe_filename = f"{video_name.replace(' ', '_')}_{timestamp}_{unique_id}{file_extension}"
    
    file_path = UPLOAD_DIR / safe_filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)
    
    timestamps = []
    result = insert_video(
        video_name=video_name,
        file_path=str(file_path),
        timestamps_json=timestamps
    )
    
    return {
        "id": str(result),
        "video_name": video_name,
        "file_path": str(file_path),
        "file_url": f"/videos/{safe_filename}",
        "message": "Video successfully uploaded and added to database"
    }
    return {"hello": 1}
    

def insert_video(video_name, file_path, timestamps_json):
    video_document = {
        "video_name": video_name,
        "file_path": file_path,
        "timestamps": timestamps_json
    }
    
    result = videos_collection.insert_one(video_document)
    return result.inserted_id

@app.get("/api/videos/")
async def get_videos():
    videos = list(videos_collection.find({}, {"_id": 1, "video_name": 1, "file_path": 1, "timestamps": 1}))
    
    for video in videos:
        video["_id"] = str(video["_id"])
        
        filename = os.path.basename(video["file_path"])
        video["url"] = f"/static/videos/{filename}"
    
    return videos

@app.get("/api/videos/{video_id}")
async def get_video(video_id: str):
    """
    Get details about a specific video by ID
    """
    from bson.objectid import ObjectId
    
    video = videos_collection.find_one({"_id": ObjectId(video_id)})
    
    if not video:
        return {"error": "Video not found"}, 404
    
    video["_id"] = str(video["_id"])
    
    filename = os.path.basename(video["file_path"])
    video["url"] = f"/static/videos/{filename}"
    
    return video


@app.get("/api/compare-videos/")
async def compare_videos(video_id1: str, video_id2: str):
    print("doing smth")
    from bson.objectid import ObjectId
    
    video1 = videos_collection.find_one({"_id": ObjectId(video_id1)})
    video2 = videos_collection.find_one({"_id": ObjectId(video_id2)})
    
    if not video1 or not video2:
        return {"error": "One or both videos not found"}, 404
    
    video1["_id"] = str(video1["_id"])
    video2["_id"] = str(video2["_id"])
    
    random_timestamps = []
    event_types = ["start", "takeoff", "landing", "stop", "turn", "acceleration"]
    
    num_events = random.randint(5, 10)
    
    for i in range(num_events):
        event_name = random.choice(event_types)
        start_time = f"{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"
        end_time = f"{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"
        
        random_timestamps.append({
            "event": f"{event_name} {i+1}",
            "start": start_time,
            "end": end_time
        })
    
    video1["timestamps"] = random_timestamps
    video2["timestamps"] = random_timestamps
    
    videos_collection.update_one(
        {"_id": ObjectId(video_id1)},
        {"$set": {"timestamps": random_timestamps}}
    )
    
    videos_collection.update_one(
        {"_id": ObjectId(video_id2)},
        {"$set": {"timestamps": random_timestamps}}
    )

    filename1 = os.path.basename(video1["file_path"])
    filename2 = os.path.basename(video2["file_path"])
    
    video1["url"] = f"/static/videos/{filename1}"
    video2["url"] = f"/static/videos/{filename2}"
    
    return {
        "video1": video1,
        "video2": video2,
        "message": "Random timestamps generated and saved to database"
    }

