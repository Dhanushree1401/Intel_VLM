The project is an AI-powered image search engine built using FastAPI and OpenAI's CLIP model. 
Here is the live link:https://dhanushree1401.github.io/Intel_VLM/

The system enables users to:
Classify images by determining the most relevant category.
Search for similar images based on classification.
Perform text-based image searches by querying descriptions in the dataset.
The core functionality is powered by OpenAI's CLIP (Contrastive Language-Image Pretraining) model, which can understand both images and text. It uses a pre-trained ViT-B-32 model to extract meaningful features and compare images with text-based queries.
The application supports image uploads via FastAPI, processes them using PIL (Pillow), and returns classification results or similar images based on a predefined dataset (image_search_dataset.json). The dataset stores categorized images with metadata, enabling both content-based and text-based searches.

Tech Stack
Backend
FastAPI: High-performance web framework for handling API requests.
OpenAI CLIP (ViT-B-32): Pretrained model for image classification and retrieval.
Torch (PyTorch): Deep learning framework to process image embeddings.
Pillow (PIL): Image processing library for handling uploads.
JSON: Stores the dataset of categorized images and their metadata.

Other Technologies
CORS Middleware: Ensures cross-origin API access.
Logging: Error handling and debugging support.
Uvicorn: ASGI server to run FastAPI applications.

Key Features
✅ Image Classification: Upload an image and classify it into predefined categories.
✅ Similar Image Search: Find similar images based on the uploaded image.
✅ Text-Based Search: Search images using textual descriptions.
✅ Fast & Scalable: Optimized API endpoints using FastAPI.
✅ CORS Enabled: API can be accessed from different frontend applications.

The setup makes it useful for image-based search engines, e-commerce applications, and AI-powered content recommendation systems. 🚀
