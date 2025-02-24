# Projet M2 CI/CD
*Wladimir LUCET*

> Accessible online: [https://projet.dobial.com](https://projet.dobial.com)

# Run the stack
1. Run `docker compose build`
2. Run `docker compose up -d`
3. Go to the url in the logs of the frontend (should be `http://localhost:8500` on the local machine and `http://frontend_project_CICD:8500` on the "frontend" network)

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
Bonus : This project used conventionnal commits, development libraries inside uv and ONNX

# Structure
## Frontend
Simple app that allows the user to drop an image file, run YOLOv11-nano and visualise the results.
The user must upload an image, then click on "Run inference".
Once inference has ended, labels and bounding boxes will be added to a new image.
- Slim python 3.12
- Listening ports : 8500, mapped ports : 8500 (we have to map the ports of the docker to make it listen on the machine instead of just inside the container itself).

## Backend
### Rest API
Rest API running on Fast API that connects the frontend to the inference container.
- Slim python 3.12
- Listening ports : 9000, mapped ports : None (there is no need to map the ports of the docker and bypass firewall limitations, as the other services can access it from the private network).

### Inference
Python app running an ONNX version of YOLOv11.
- Python 3.12 (not slim because of libgl1)
- Listening ports : 8000, mapped ports : None (there is no need to map the ports of the docker and bypass firewall limitations, as the rest api can access it from the private network).
