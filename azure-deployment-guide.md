# Azure Container Apps Deployment Guide

## Prerequisites
- Azure account with active subscription
- Azure CLI installed: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
- Docker installed (for local testing)

## Step 1: Login to Azure
```bash
az login
```

## Step 2: Set Variables
```bash
RESOURCE_GROUP="skillquest-rg"
LOCATION="eastus"
ACR_NAME="skillquestacr"
CONTAINER_APP_NAME="skillquest-rl-api"
CONTAINER_APP_ENV="skillquest-env"
```

## Step 3: Create Resource Group
```bash
az group create --name $RESOURCE_GROUP --location $LOCATION
```

## Step 4: Create Azure Container Registry
```bash
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true
```

## Step 5: Build and Push Docker Image
```bash
az acr build --registry $ACR_NAME --image skillquest-rl-api:v1 .
```

## Step 6: Create Container Apps Environment
```bash
az containerapp env create \
  --name $CONTAINER_APP_ENV \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION
```

## Step 7: Get ACR Credentials
```bash
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)
```

## Step 8: Deploy Container App
```bash
az containerapp create \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_APP_ENV \
  --image $ACR_NAME.azurecr.io/skillquest-rl-api:v1 \
  --target-port 8000 \
  --ingress 'external' \
  --registry-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --cpu 1.0 \
  --memory 2.0Gi \
  --min-replicas 1 \
  --max-replicas 3
```

## Step 9: Get Your API URL
```bash
az containerapp show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn -o tsv
```

Your API will be available at: `https://<output-from-above>`

## Testing Your API

### Health Check
```bash
curl https://your-app-url.azurecontainerapps.io/health
```

### Get Action (example)
```bash
curl -X POST https://your-app-url.azurecontainerapps.io/get_action \
  -H "Content-Type: application/json" \
  -d '{
    "total_xp": 500,
    "rank_percentile": 0.25,
    "badges_earned": 3,
    "avg_score": 75,
    "days_since_login": 2,
    "streak_days": 5,
    "tasks_completed": 20,
    "is_struggling": false
  }'
```

## Updating Your API

When you make changes:

```bash
# Build and push new version
az acr build --registry $ACR_NAME --image skillquest-rl-api:v2 .

# Update the container app
az containerapp update \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --image $ACR_NAME.azurecr.io/skillquest-rl-api:v2
```

## Cost Optimization

- **Free tier**: Container Apps includes free grant for compute and requests
- **Auto-scaling**: Set `--min-replicas 0` to scale to zero when not in use
- **Monitor costs**: Use Azure Cost Management in the portal

## Troubleshooting

### View logs
```bash
az containerapp logs show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --follow
```

### Check revision status
```bash
az containerapp revision list \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  -o table
```

## Additional Resources

- [Azure Container Apps Documentation](https://learn.microsoft.com/en-us/azure/container-apps/)
- [Azure Student Credits](https://azure.microsoft.com/en-us/free/students/)
- [Azure Container Registry](https://learn.microsoft.com/en-us/azure/container-registry/)
