import kagglehub

# Download latest version
path = kagglehub.dataset_download("stackoverflow/stacksample")

print("Path to dataset files:", path)