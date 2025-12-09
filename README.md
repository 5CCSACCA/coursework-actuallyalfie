# Cloud-Based Multimodal AI System  
YOLO, RabbitMQ, BitNet, Firebase, MongoDB

Module: Cloud Computing for Artificial Intelligence (5CCSACCA)  
Student: Alfie Pike  
Repository: https://github.com/5CCSACCA/coursework-actuallyalfie  

## Project Overview

This project implements a cloud-based multimodal AI system using a microservices architecture. The system performs object detection on uploaded images and generates natural-language descriptions of the detected content. Services communicate asynchronously using a message queue and are deployed using Docker.

The design focuses on scalability, separation of concerns, and fault tolerance, following event-driven and cloud-native principles covered in the module.

## System Architecture

User  
→ FastAPI (YOLO Service)  
→ Object Detection (YOLO)  
→ RabbitMQ  
→ BitNet Language Model Service  
→ MongoDB and Firebase  
→ Results retrieved via API  

Each component runs as an independent container and communicates only through defined interfaces.

## Processing Workflow

1. A user uploads an image to the YOLO FastAPI endpoint.  
2. YOLO performs object detection and outputs bounding boxes and confidence scores.  
3. Detection metadata is published to a RabbitMQ queue.  
4. The BitNet service consumes the message and generates a textual description.  
5. Results are stored in MongoDB and Firebase.  
6. The user retrieves the generated output via the API.

## Technologies Used

- Python 3  
- FastAPI  
- Ultralytics YOLO  
- Microsoft BitNet (GGUF-quantized LLM)  
- RabbitMQ  
- MongoDB  
- Firebase Authentication and Storage  
- Docker and Docker Compose  
- Prometheus and Grafana  
- Pytest  

## Authentication and Security

User authentication is implemented using Firebase Authentication.

Users register and log in using the `/auth/register` and `/auth/login` endpoints. After authentication, Firebase issues an ID token which must be included in requests to protected endpoints using the following header:

Authorization: Bearer <ID_TOKEN>

Requests without a valid token return HTTP 401 Unauthorized.  
Sensitive credentials are stored using environment variables and are not hardcoded in the repository.

## Monitoring and Observability

Prometheus is used to collect metrics from the YOLO and BitNet services.  
Grafana dashboards visualise request rates, service activity, and inference behaviour.

## Automated Testing

Unit tests are implemented using Pytest.

All tests can be executed using:

```bash
./run_tests.sh
```

This script installs development dependencies from `requirements-dev.txt` and runs the full test suite.

## Deployment with Docker

Prerequisites:
- Docker  
- Docker Compose  

To start the full system:

```bash
sudo docker compose up --build -d
```

Service endpoints:

- YOLO API: http://localhost:8888/docs  
- BitNet API: http://localhost:9999/docs  
- Grafana: http://localhost:3000  
- Prometheus: http://localhost:9090  
- RabbitMQ Management UI: http://localhost:15672  

Default RabbitMQ credentials:  
Username: guest  
Password: guest  

## One-Command Startup Script

A helper script is included to simplify system startup and model setup.

The script:
- Checks it is being run from the repository root  
- Downloads the BitNet GGUF model if it is not already present  
- Places the model in `bitnet_cpp/model/`  
- Starts all services using Docker Compose  

To run the script:

```bash
chmod +x start_system.sh
./start_system.sh
```

YOLO model weights are downloaded automatically by Ultralytics on first use.

## Cost Estimation

Cost modelling follows the formula:

Total Cost = C_YOLO + C_BitNet + C_RabbitMQ + (C_Workers × N) + F  

Where:
- C_YOLO represents YOLO inference compute cost  
- C_BitNet represents BitNet inference compute cost  
- C_RabbitMQ represents the message broker cost  
- N is the number of scaled worker nodes  
- F is a constant Firebase storage cost  

For an estimated 200,000 concurrent users, both YOLO and BitNet are assumed to scale linearly using one virtual machine per 100 users.

## Project Structure

```bash
.
├── yolo_service/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── bitnet_service/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── docker-compose.yml
├── prometheus.yml
├── run_tests.sh
├── requirements-dev.txt
├── README.md
└── start_system.sh
```

## Video Demonstration

The submitted video demonstrates:
- Overall system architecture  
- YOLO image detection  
- RabbitMQ message flow  
- BitNet text generation  
- Firebase authentication  
- Data storage using MongoDB and Firebase  
- Monitoring with Prometheus and Grafana  
- Cost estimation approach  
- Implemented security measures  

## Coursework Progress

Stage 1–6: Completed  
Stage 7 (Authentication): Completed  
Stage 8 (Cost Modelling): Completed  
Stage 9 (Monitoring): Completed  
Stage 10 (Testing): Completed  
Stage 11 (Security): Completed  
