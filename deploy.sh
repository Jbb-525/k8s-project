#!/bin/bash

# Deploy MNIST application to Kubernetes cluster

set -e  # Exit immediately on error

echo "================================================"
echo "Deploying MNIST Application to Kubernetes"
echo "================================================"

cd k8s-configs

# 1. Create persistent storage
echo ""
echo "[1/4] Creating persistent storage (PVC)..."
kubectl apply -f pvc.yaml
echo "PVC created"

# Wait for PVC to bind
echo "Waiting for PVC to bind..."
kubectl wait --for=condition=Bound pvc/model-storage-pvc --timeout=60s || echo "⚠ PVC may still be pending"

# 2. Run training Job
echo ""
echo "[2/4] Starting training job..."
kubectl apply -f training-job.yaml
echo "Training Job created"

# Wait for training to complete
echo ""
echo "Waiting for training job to complete..."
echo "(This may take several minutes)"
kubectl wait --for=condition=complete job/mnist-training-job --timeout=1200s || {
    echo "Training job timed out or failed"
    echo "View logs:"
    kubectl logs -l component=training --tail=50
    exit 1
}
echo "Training completed"

# Show training logs
echo ""
echo "Training logs:"
kubectl logs -l component=training --tail=20

# 3. Deploy inference service
echo ""
echo "[3/4] Deploying inference service..."
kubectl apply -f inference-deployment.yaml
kubectl apply -f inference-service.yaml
echo "Inference service deployed"

# Wait for Deployment to be ready
echo ""
echo "Waiting for inference service to be ready..."
kubectl wait --for=condition=available deployment/mnist-inference --timeout=300s
echo "Inference service is ready"

# 4. Get service access information
echo ""
echo "[4/4] Getting service access information..."
echo ""
echo "================================================"
echo "Deployment Complete!"
echo "================================================"
echo ""
echo "Service information:"
kubectl get service mnist-inference-service
echo ""
echo "Pod status:"
kubectl get pods -l app=mnist
echo ""
echo "Waiting for external IP to be assigned..."
echo "(This may take 1-2 minutes)"
echo ""

# Wait for and display external IP
EXTERNAL_IP=""
while [ -z "$EXTERNAL_IP" ]; do
    EXTERNAL_IP=$(kubectl get service mnist-inference-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    if [ -z "$EXTERNAL_IP" ]; then
        echo "Waiting for external IP assignment..."
        sleep 10
    fi
done

echo ""
echo "================================================"
echo "Deployment Successful!"
echo "================================================"
echo ""
echo "Access URL: http://$EXTERNAL_IP"
echo ""
echo "Useful commands:"
echo "  View Pod status:      kubectl get pods -l app=mnist"
echo "  View Service status:  kubectl get service mnist-inference-service"
echo "  View training logs:   kubectl logs -l component=training"
echo "  View inference logs:  kubectl logs -l component=inference"
echo "  Delete all resources: kubectl delete -f ."
echo ""