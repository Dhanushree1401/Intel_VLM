The project is an AI-powered image search engine that was constructed with the CLIP model from OpenAI and FastAPI.

By identifying the most appropriate category, the system allows users to: Classify images. Use classification to find related images. Use the dataset's descriptions to conduct text-based image searches. OpenAI's CLIP (Contrastive Language-Image Pretraining) model, which is capable of comprehending both text and images, powers the essential features. It compares images using text-based queries and extracts significant features using a pre-trained ViT-B-32 model. The program accepts FastAPI image uploads, uses Pillow (PIL) to process the images, and then uses a predefined dataset (image_search_dataset.json) to return classification results or similar images. Both content-based and text-based searches are made possible by the dataset's storage of categorized images with metadata.

Backend Tech Stacks are FastAPI: A web framework with high performance for managing API requests. OpenAI CLIP (ViT-B-32): A pre-trained model for retrieving and classifying images. Torch (PyTorch): A deep learning framework for processing embeddings in images. Pillow (PIL): A library for image processing that manages uploads. JSON: Holds the metadata and image classification dataset.

Additional technologies include CORS Middleware, which guarantees access to cross-origin APIs. Logging: Support for debugging and error handling. Uvicorn: ASGI server for FastAPI applications.

Important Features Classification of Images: Upload an image and assign it to one of the pre-established categories. Similar Image Search: Use the uploaded image to find related images. Text-Based Search: Use textual descriptions to look for images. Fast & Scalable: FastAPI was used to optimize API endpoints. When CORS is enabled, various frontend applications can access the API.

Because of the configuration, it can be used with e-commerce apps, image-based search engines, and AI-powered content recommendation systems. 🚀

