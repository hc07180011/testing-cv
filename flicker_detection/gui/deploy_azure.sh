az staticwebapp create \
    -n NTU-Google-Testing \
    -g sqai_group \
    -s https://github.com/hc07180011/NTU-Google-Testing \
    -l EastAsia \
    -b main \
    --app-location "build" \
    --token <YOUR_GITHUB_PERSONAL_ACCESS_TOKEN>

# az staticwebapp delete \
#     --name NTU-Google-Testing \
#     --resource-group sqai_group