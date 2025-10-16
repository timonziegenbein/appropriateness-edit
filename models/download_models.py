from huggingface_hub import snapshot_download

snapshot_download(repo_id="distilbert/distilgpt2", repo_type="model")
snapshot_download(repo_id="microsoft/deberta-xlarge-mnli", repo_type="model")
snapshot_download(repo_id="meta-llama/Llama-3.1-8B-Instruct", repo_type="model")
snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", repo_type="model")

snapshot_download(repo_id="timonziegenbein/appropriateness-corpus-extension", repo_type="dataset")
snapshot_download(repo_id="timonziegenbein/appropriateness-corpus", repo_type="dataset")
