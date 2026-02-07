# Azure Deploy and Test (Engagement-only Feedback)

## Build and update
```cmd
cd "D:\My Projects\skillquest-rl-api"
az acr build --registry %ACR_NAME% --image skillquest-rl-api:v6 .
az containerapp update --name %CONTAINER_APP_NAME% --resource-group %RESOURCE_GROUP% --image %ACR_NAME%.azurecr.io/skillquest-rl-api:v6
```

## Optional: API key for feedback/save
```cmd
az containerapp secret set --name %CONTAINER_APP_NAME% --resource-group %RESOURCE_GROUP% --secrets API_KEY=YourStrongSecretHere
az containerapp update --name %CONTAINER_APP_NAME% --resource-group %RESOURCE_GROUP% --set-env-vars API_KEY=secretref:API_KEY
```

## Test predict → get recommendation_id
```cmd
curl -X POST https://skillquest-rl-api.livelytree-4b213315.eastasia.azurecontainerapps.io/predict -H "Content-Type: application/json" -d "{\"user_id\":1001,\"level\":\"Beginner\",\"total_badges\":0,\"quiz_score\":40,\"quiz_accuracy\":0.40,\"recent_points\":50,\"daily_xp\":10,\"days_since_last_login\":7,\"session_duration\":60,\"active_minutes\":5,\"modules_done\":0,\"consecutive_completions\":0}"
```

## Send feedback ONLY when engaged
```cmd
curl -X POST https://skillquest-rl-api.livelytree-4b213315.eastasia.azurecontainerapps.io/feedback -H "Content-Type: application/json" -H "X-API-Key: YourStrongSecretHere" -d "{\"recommendation_id\":\"REPLACE_ID\",\"engaged\":true}"
```

If no engagement happens within 12 hours, DO NOT send feedback—the API will auto-penalize the recommendation.

## Health and stats
```cmd
curl https://skillquest-rl-api.livelytree-4b213315.eastasia.azurecontainerapps.io/health
curl https://skillquest-rl-api.livelytree-4b213315.eastasia.azurecontainerapps.io/stats
```

## Faster timeout testing (optional)
Temporarily set a tiny TIMEOUT_HOURS (e.g., 0.003 ~= 10 seconds):
- In Azure Portal → your Container App → Environment Variables → TIMEOUT_HOURS=0.003, then Redeploy.
- Or CLI: `az containerapp update --name ... --set-env-vars TIMEOUT_HOURS=0.003`