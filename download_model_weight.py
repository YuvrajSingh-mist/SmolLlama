import gdown
import os
import argparse

def download_model(model_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    url = f"https://drive.google.com/uc?id={model_id}"
    output_path = os.path.join(folder, filename)
    print(f"Downloading model to {output_path}...")
    gdown.download(url, output_path, quiet=False)
    print("Download complete!")

def main():
    parser = argparse.ArgumentParser(description="Download models using gdown and organize them into appropriate folders.")
    parser.add_argument("-P", "--pretrained", action="store_true", help="Download the pretrained model")
    parser.add_argument("-F", "--finetuned", action="store_true", help="Download the fine-tuned model")
    parser.add_argument("-D", "--dpo", action="store_true", help="Download the DPO model")
    
    args = parser.parse_args()
    
    pretrained_model_file_id = "1efKljK-cnM1NuF2Mgx86BljQuEI1B4Em"
    fine_tuned_model_id = "1Cz7cZnRjo6x9m-axQxzEO2cEo1-z4mxx"
    dpo_model_file_id = "1pporxPFIdtL3sDCstsRhJaiXuvkR45SC"
    
    if args.pretrained:
        download_model(pretrained_model_file_id, "weights/pretrained", "pretrained_model.pt")
    if args.finetuned:
        download_model(fine_tuned_model_id, "weights/fine_tuned", "fine_tuned_model.pt")
    if args.dpo:
        download_model(dpo_model_file_id, "weights/DPO", "dpo_model.pt")

if __name__ == "__main__":
    main()