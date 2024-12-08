# Create a PowerShell version (deploy.ps1)
$PROJECT_ID="prj-6-400921"
$IMAGE_NAME="audio-backend"
$REGION="us-central1"
$MEMORY="2Gi"
$CPU="2"

Write-Host "Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME .

Write-Host "Pushing image to Google Container Registry..."
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME

Write-Host "Deploying to Cloud Run..."
gcloud run deploy $IMAGE_NAME `
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME `
  --platform managed `
  --region $REGION `
  --allow-unauthenticated `
  --memory $MEMORY `
  --cpu $CPU

Write-Host "Deployment completed!"