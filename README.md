# MNIST Kubernetes Deployment

A complete machine learning deployment project that trains and serves a handwritten digit recognition model using PyTorch, Docker, and Kubernetes on Google Cloud Platform.

## 🎯 Project Overview

This project demonstrates:
- Training a CNN model on MNIST dataset using PyTorch
- Containerizing training and inference services with Docker
- Deploying to Google Kubernetes Engine (GKE) with self-healing capabilities
- Providing a web interface and REST API for digit recognition

**Model Accuracy**: 98.95% on MNIST test set

## 🏗️ Architecture

```
Training Job → Saves Model → PersistentVolume
                                    ↓
Inference Pods (2 replicas) ← Load Model
                ↓
LoadBalancer Service → External Access
```

## 🚀 Quick Start

### Prerequisites

- Google Cloud Platform account
- `gcloud` CLI installed
- `kubectl` installed
- Docker installed

### 1. Setup GCP

```bash
# Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 2. Create GKE Cluster

```bash
gcloud container clusters create mnist-cluster \
    --zone us-central1-a \
    --num-nodes 2 \
    --machine-type e2-standard-2

# Get credentials
gcloud container clusters get-credentials mnist-cluster --zone us-central1-a
```

### 3. Build and Push Docker Images

```bash
./build.sh
# Enter your GCP project ID when prompted
```

### 4. Deploy to Kubernetes

```bash
./deploy.sh
```

The script will:
- Create persistent storage (1GB)
- Run training job (~5 minutes)
- Deploy inference service (2 replicas)
- Display external IP for access

### 5. Access the Service

```bash
# Get external IP
kubectl get service mnist-inference-service

# Open in browser
http://EXTERNAL_IP
```

## 📁 Project Structure

```
mnist-k8s-project/
├── training/
│   ├── train.py              # PyTorch training script
│   ├── Dockerfile            # Training container
│   └── requirements.txt
├── inference/
│   ├── app.py                # Flask inference service
│   ├── Dockerfile            # Inference container
│   └── requirements.txt
├── k8s-configs/
│   ├── pvc.yaml              # Persistent storage
│   ├── training-job.yaml     # Training job
│   ├── inference-deployment.yaml  # Inference deployment
│   └── inference-service.yaml     # LoadBalancer service
├── build.sh                  # Build & push images
└── deploy.sh                 # Deploy to K8s
```

## 🔧 Key Features

### Self-Healing Mechanisms

- **Liveness Probe**: Automatically restarts unhealthy containers
- **Readiness Probe**: Removes unready pods from load balancer
- **ReplicaSet**: Maintains 2 pod replicas with automatic replacement
- **Node Failure Recovery**: Reschedules pods if nodes fail

### Technology Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | PyTorch 2.1.0 |
| Web Framework | Flask 3.0.0 |
| Containerization | Docker |
| Orchestration | Kubernetes (GKE) |
| Cloud Platform | Google Cloud Platform |

## 📊 Usage Examples

### Web Interface

Upload a handwritten digit image and get instant predictions with confidence scores.

### REST API

```bash
# Predict digit from image
curl -X POST http://EXTERNAL_IP/predict \
  -F "file=@digit.png"

# Response
{
  "prediction": 7,
  "confidence": 0.9876,
  "all_predictions": [0.001, 0.002, ..., 0.9876, ...]
}

# Health check
curl http://EXTERNAL_IP/health
```

## 🛠️ Useful Commands

```bash
# View all pods
kubectl get pods -l app=mnist

# View training logs
kubectl logs -l component=training

# View inference logs
kubectl logs -l component=inference

# View service status
kubectl get service mnist-inference-service

# Delete all resources
cd k8s-configs && kubectl delete -f .
```

## 🧹 Cleanup

```bash
# Delete Kubernetes resources
cd k8s-configs
kubectl delete -f .

# Delete GKE cluster
gcloud container clusters delete mnist-cluster --zone us-central1-a

# Delete Docker images from GCR
gcloud container images delete gcr.io/YOUR_PROJECT_ID/mnist-training:latest
gcloud container images delete gcr.io/YOUR_PROJECT_ID/mnist-inference:latest
```

## 📝 Notes

- No GPU required - runs on CPU instances
- Training takes approximately 5 minutes
- First-time image build may take 10-15 minutes
- Subsequent builds are faster due to layer caching

## 📚 Assignment

This project is completed as part of **Homework 4: AI Service in Container** coursework, demonstrating containerized ML deployment with Kubernetes orchestration.

## 📄 License

Educational project for academic purposes.
