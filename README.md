# Projet M2 CI/CD
*Wladimir LUCET*

> Accessible online: [https://projet.dobial.com](https://projet.dobial.com)

# Run the stack
1. Run `docker compose build`
2. Run `docker compose up -d`
3. Go to the url in the logs of the frontend (should be `http://frontend_project_CICD:8500` on the "frontend" network)

# CI/CD
1. Pre-commits
3. Github Actions
   1. Run tests
   2. Create images
   3. Send images to container registry
4. Docker compose
   1. Shared networks
   2. Dependence on other services
   3. Environment variables

# Structure
## Frontend
Simple app that allows the user to drop an image file, run YOLOv11-nano and visualise the results.
The user must upload an image, then click on "Run inference".
Once inference has ended, labels and bounding boxes will be added to a new image.
- Slim python 3.12

## Backend
### Rest API
Rest API running on Fast API that connects the frontend to the inference container.
- Slim python 3.12

### Inference
Python app running an ONNX version of YOLOv11.
- Python 3.12 (not slim because of libgl1)
