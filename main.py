import os
import json
import logging
from typing import List, Dict, Any

import torch
import open_clip
import requests
from PIL import Image
import io
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageSearchEngine:
    def __init__(self, dataset_path: str = 'image_search_dataset.json'):
        """
        Initialize image search engine with CLIP model and dataset
        
        Args:
            dataset_path (str): Path to the image dataset
        """
        try:
            # Load dataset
            with open(dataset_path, 'r') as f:
                self.dataset = json.load(f)
            
            # Load CLIP Model
            self.model, self.preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            self.model, self.preprocess, self.tokenizer = None, None, None
            self.dataset = {}

    def classify_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify an image using CLIP model
        
        Args:
            image (PIL.Image): Input image
        
        Returns:
            Dict with classification results
        """
        if not self.model or not self.preprocess or not self.tokenizer:
            raise HTTPException(status_code=500, detail="CLIP model not loaded")

        try:
            # Predefined categories for classification
            categories = list(self.dataset.keys())

            # Process image for CLIP
            image_tensor = self.preprocess(image).unsqueeze(0)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                text_inputs = self.tokenizer(categories)
                text_features = self.model.encode_text(text_inputs)

            # Compute similarities
            similarities = (image_features @ text_features.T).softmax(dim=-1)
            top_predictions = sorted(
                zip(categories, similarities.squeeze().tolist()), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]

            return {
                "top_predictions": [
                    {"category": cat, "confidence": float(conf)} 
                    for cat, conf in top_predictions
                ]
            }
        
        except Exception as e:
            logger.error(f"Classification error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def search_similar_images(self, image: Image.Image) -> Dict[str, Any]:
        """
        Search for similar images based on classification
        
        Args:
            image (PIL.Image): Input image
        
        Returns:
            Dict with similar images
        """
        try:
            # First, classify the image
            classification = self.classify_image(image)
            top_category = classification['top_predictions'][0]['category']

            # Search similar images from dataset
            similar_images = self.dataset.get(top_category, [])
            
            return {
                "category": top_category,
                "classification": classification,
                "similar_images": similar_images
            }
        
        except Exception as e:
            logger.error(f"Similar image search error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def text_based_image_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform text-based image search across dataset
        
        Args:
            query (str): Search text
        
        Returns:
            List of matching images
        """
        try:
            # Perform case-insensitive search across dataset
            search_results = []
            
            for category, images in self.dataset.items():
                # Check if query matches category
                if query.lower() in category.lower():
                    search_results.extend(images)
                
                # Check descriptions if available
                for image in images:
                    if 'description' in image and query.lower() in image['description'].lower():
                        search_results.append(image)
            
            return search_results
        
        except Exception as e:
            logger.error(f"Text-based search error: {e}")
            return []

# FastAPI Application
app = FastAPI(title="Comprehensive Image Search")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Image Search Engine
image_search_engine = ImageSearchEngine()

@app.post("/classify-image")
async def classify_image(image: UploadFile = File(...)):
    """
    Classify uploaded image
    """
    try:
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        results = image_search_engine.classify_image(image)
        return JSONResponse(content=results)
    
    except HTTPException as http_err:
        return JSONResponse(status_code=http_err.status_code, content={"error": str(http_err.detail)})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

@app.post("/search-similar-images")
async def search_similar_images(image: UploadFile = File(...)):
    """
    Search for similar images based on uploaded image
    """
    try:
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        results = image_search_engine.search_similar_images(image)
        return JSONResponse(content=results)
    
    except HTTPException as http_err:
        return JSONResponse(status_code=http_err.status_code, content={"error": str(http_err.detail)})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

@app.get("/search-images-by-text")
async def search_images_by_text(query: str = Query(..., description="Search text for images")):
    """
    Perform text-based image search
    """
    try:
        results = image_search_engine.text_based_image_search(query)
        return JSONResponse(content={"images": results})
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
