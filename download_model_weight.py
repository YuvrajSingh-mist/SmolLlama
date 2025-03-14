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


import os
import argparse

def download_model(model_id, folder, filename, access_token):
    os.makedirs(folder, exist_ok=True)
    output_path = os.path.join(folder, filename)
    
    url = f"https://www.googleapis.com/drive/v3/files/{model_id}?alt=media"
    command = f"curl -H \"Authorization: Bearer {access_token}\" {url} -o {output_path}"
    
    print(f"Downloading model to {output_path}...")
    os.system(command)
    print("Download complete!")

def main():
    parser = argparse.ArgumentParser(description="Download models using Google Drive API and organize them into appropriate folders.")
    parser.add_argument("-P", "--pretrained", action="store_true", help="Download the pretrained model")
    parser.add_argument("-F", "--sft", action="store_true", help="Download the fine-tuned model")
    parser.add_argument("-D", "--dpo", action="store_true", help="Download the DPO model")
    parser.add_argument("--token", type=str, required=True, help="Google Drive API Access Token")
    
    args = parser.parse_args()
    
    pretrained_model_file_id = "1CwtDjbN6a7tt7mykywxAANHBTvdSr-98"
    fine_tuned_model_id = "10bsea7_MFXw6T967iCrp6zSGMfqDljHf"
    dpo_model_file_id = "1hIzV_VVdvmplQQuaH9QQCcmUbfolFjyh"
    
    if args.pretrained:
        download_model(pretrained_model_file_id, "weights/pretrained", "pretrained_model.pt", args.token)
    if args.sft:
        download_model(fine_tuned_model_id, "weights/fine_tuned", "fine_tuned_model.pt", args.token)
    if args.dpo:
        download_model(dpo_model_file_id, "weights/DPO", "dpo_model.pt", args.token)

if __name__ == "__main__":
    main()

