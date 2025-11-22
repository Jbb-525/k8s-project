#!/bin/bash

# Build and push Docker images to Google Container Registry
# Please set your GCP project ID before running

set -e  # Exit immediately on error

echo "================================================"
echo "Building and Pushing MNIST Docker Images"
echo "================================================"

# 1. Set your GCP project ID
read -p "Please enter your GCP project ID: " PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    echo "Error: Project ID cannot be empty"
    exit 1
fi

echo "Using project ID: $PROJECT_ID"

# 2. Configure Docker authentication
echo ""
echo "[1/4] Configuring Docker authentication to GCR..."
gcloud auth configure-docker

# 3. Build training image
echo ""
echo "[2/4] Building training image..."
cd training
docker build -t gcr.io/$PROJECT_ID/mnist-training:latest .
echo "Training image built successfully"

# 4. Build inference image
echo ""
echo "[3/4] Building inference image..."
cd ../inference
docker build -t gcr.io/$PROJECT_ID/mnist-inference:latest .
echo "Inference image built successfully"

# 5. Push images to GCR
echo ""
echo "[4/4] Pushing images to Google Container Registry..."
docker push gcr.io/$PROJECT_ID/mnist-training:latest
docker push gcr.io/$PROJECT_ID/mnist-inference:latest
echo "Images pushed successfully"

# 6. Update Kubernetes configuration files
echo ""
echo "Updating Kubernetes configuration files with image addresses..."
cd ../k8s-configs

# Use sed to replace placeholder
sed -i.bak "s|gcr.io/YOUR_PROJECT_ID|gcr.io/$PROJECT_ID|g" training-job.yaml
sed -i.bak "s|gcr.io/YOUR_PROJECT_ID|gcr.io/$PROJECT_ID|g" inference-deployment.yaml

# Delete backup files
rm -f *.bak

echo "Configuration files updated"

echo ""
echo "================================================"
echo "All images built and pushed successfully!"
echo "================================================"
echo ""
echo "Image addresses:"
echo "  Training: gcr.io/$PROJECT_ID/mnist-training:latest"
echo "  Inference: gcr.io/$PROJECT_ID/mnist-inference:latest"
echo ""
echo "Next steps:"
echo "  1. Create GKE cluster (if not already created)"
echo "  2. Run ./deploy.sh to deploy to Kubernetes"