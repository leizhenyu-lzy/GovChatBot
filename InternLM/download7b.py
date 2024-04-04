from huggingface_hub import hf_hub_download

REPO_NAME = "internlm/internlm2-chat-7b"
FILE_NAME = "config.json"

filePath = hf_hub_download(repo_id=REPO_NAME, 
                           filename=FILE_NAME)

print("filePath - ", filePath)

