# import gdown
# import os
# import argparse

# def download_model(model_id, folder, filename):
#     os.makedirs(folder, exist_ok=True)
#     url = f"https://drive.google.com/uc?id={model_id}"
#     output_path = os.path.join(folder, filename)
#     print(f"Downloading model to {output_path}...")
#     gdown.download(url, output_path, quiet=False)
#     print("Download complete!")

# def main():
#     parser = argparse.ArgumentParser(description="Download models using gdown and organize them into appropriate folders.")
#     parser.add_argument("-P", "--pretrained", action="store_true", help="Download the pretrained model")
#     parser.add_argument("-F", "--sft", action="store_true", help="Download the fine-tuned model")
#     parser.add_argument("-D", "--dpo", action="store_true", help="Download the DPO model")
    
#     args = parser.parse_args()
    
#     pretrained_model_file_id = "1CwtDjbN6a7tt7mykywxAANHBTvdSr-98"
#     fine_tuned_model_id = "10bsea7_MFXw6T967iCrp6zSGMfqDljHf"
#     dpo_model_file_id = "1hIzV_VVdvmplQQuaH9QQCcmUbfolFjyh"
    
#     if args.pretrained:
#         download_model(pretrained_model_file_id, "weights/pretrained", "pretrained_model.pt")
#     if args.sft:
#         download_model(fine_tuned_model_id, "weights/fine_tuned", "fine_tuned_model.pt")
#     if args.dpo:
#         download_model(dpo_model_file_id, "weights/DPO", "dpo_model.pt")

# if __name__ == "__main__":
#     main()


# import os
# import argparse

# def download_model(model_id, folder, filename, access_token):
#     os.makedirs(folder, exist_ok=True)
#     output_path = os.path.join(folder, filename)
    
#     url = f"https://www.googleapis.com/drive/v3/files/{model_id}?alt=media"
#     command = f"curl -H \"Authorization: Bearer {access_token}\" {url} -o {output_path}"
    
#     print(f"Downloading model to {output_path}...")
#     os.system(command)
#     print("Download complete!")

# def main():
#     parser = argparse.ArgumentParser(description="Download models using Google Drive API and organize them into appropriate folders.")
#     parser.add_argument("-P", "--pretrained", action="store_true", help="Download the pretrained model")
#     parser.add_argument("-F", "--sft", action="store_true", help="Download the fine-tuned model")
#     parser.add_argument("-D", "--dpo", action="store_true", help="Download the DPO model")
#     parser.add_argument("--token", type=str, required=True, help="Google Drive API Access Token")
    
#     args = parser.parse_args()
    
#     pretrained_model_file_id = "1CwtDjbN6a7tt7mykywxAANHBTvdSr-98"
#     fine_tuned_model_id = "10bsea7_MFXw6T967iCrp6zSGMfqDljHf"
#     dpo_model_file_id = "1hIzV_VVdvmplQQuaH9QQCcmUbfolFjyh"
    
#     if args.pretrained:
#         download_model(pretrained_model_file_id, "weights/pretrained", "pretrained_model.pt", args.token)
#     if args.sft:
#         download_model(fine_tuned_model_id, "weights/fine_tuned", "fine_tuned_model.pt", args.token)
#     if args.dpo:
#         download_model(dpo_model_file_id, "weights/DPO", "dpo_model.pt", args.token)

# if __name__ == "__main__":
#     main()



# download_model_weight.py
import os
import argparse
from huggingface_hub import hf_hub_download, login

def download_model(repo_id, filename, cache_dir):

    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=True,
            force_download=False,
            token=os.getenv("HF_TOKEN")
        )
       
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1024*1024:
            return model_path
        raise ValueError("Downloaded file is too small or invalid")
    except Exception as e:
        print(f"Download failed: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download models from Hugging Face Hub")
    parser.add_argument("--model_type", 
                      choices=["sft", "dpo", "pretrained"],
                      required=True,
                      help="Type of model to download")
    
    args = parser.parse_args()

    model_config = {
        "sft": {
            "repo_id": "YuvrajSingh9886/smol-llama-finetuned",
            "filename": "snapshot_fine_tuned_model_900.pt",
            "cache_dir": "weights/fine_tuned"
        },
        "dpo": {
            "repo_id": "YuvrajSingh9886/smol-llama-dpo",
            "filename": "DPO_model_1650.pt",
            "cache_dir": "weights/DPO"
        },
        "pretrained": {
            "repo_id": "YuvrajSingh9886/smol-llama-base",
            "filename": "snapshot_6750.pt",
            "cache_dir": "weights/pretrained"
        }
    }

    config = model_config[args.model_type]
    os.makedirs(config["cache_dir"], exist_ok=True)
    
    print(f"Downloading {args.model_type} model...")
    model_path = download_model(
        config["repo_id"],
        config["filename"],
        config["cache_dir"]
    )
    print(f"Successfully downloaded to: {model_path}")

if __name__ == "__main__":

    login(token=os.getenv("HF_TOKEN"))
    main()
