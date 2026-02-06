from huggingface_hub import HfApi, upload_folder
import os
import yaml
import json


def create_readme(hparams_file, training_info_file, output_file, collection, description, model_name):
    """
    Creates a README.md file for the Hugging Face model card based on training info and hyperparameters.

    :param hparams_file: Path to the YAML file containing model hyperparameters.
    :param training_info_file: Path to the JSON file containing training metadata.
    :param output_file: Path to save the generated README.md file.
    :param collection: The collection name for the model.
    :param description: A description of the model.
    """
    # Load hyperparameters
    with open(hparams_file, "r") as f:
        hparams = yaml.safe_load(f)

    # Load training information
    with open(training_info_file, "r") as f:
        training_info = json.load(f)

    # Add YAML metadata block
    yaml_metadata = f"""---
library_name: pytorch
tags:
- biosignals
- {model_name.lower()}
metrics:
- validation_loss
---
"""
    # Generate README content
    readme_content = f"""# Model Card for {model_name}\n
    
- Collection: {collection}\n
- Description: {description}\n

```json
{json.dumps(training_info, indent=4)}
\n
## Hyperparameters\n
{yaml.dump(hparams, default_flow_style=False, sort_keys=False)}

# Example\n
import torch\n
from production_models import {model_name}\n
model = {model_name}()\n
signal = torch.rand(1, 100, 1)  # Example input signal\n
predictions = model.predict(signal)\n
print(predictions)\n
"""

    full_content = yaml_metadata + readme_content
    # Write README content to file
    with open(output_file, "w") as f:
        f.write(full_content)
    print(f"README.md generated and saved to: {output_file}")


def upload_production_model(local_dir, repo_name, token, model_name, description=""):
    """
    Uploads trained model files to Hugging Face Model Hub.
    :param local_dir: Directory containing model files (model_weights.pth, hparams.yaml, training_info.json).
    :param repo_name: Hugging Face repository name.
    :param token: Hugging Face authentication token.
    :param description: Model description for the README.
    """
    # Validate inputs
    if not local_dir:
        local_dir = input("Please provide the local directory containing model files: ")
    if not os.path.exists(local_dir):
        raise ValueError(f"The directory '{local_dir}' does not exist. Please provide a valid path.")

    if not repo_name:
        repo_name = input("Please provide the Hugging Face repository name: ")

    if not token:
        token = input("Please provide your Hugging Face authentication token: ")

    if not model_name:
        model_name = input("Please provide the model name: ")

    if not description:
        print("\nA model description is required. It should include:")
        print("- What the model does.")
        print("- If the model has been published, provide the reference (paper, publication,...).")
        print("- If the model has not been published, provide performance details.")
        print("- Key details about the model so others understand how to use it.")
        description = input("Please provide the model description: ")

    # Create README file
    readme_path = os.path.join(local_dir, "README.md")
    json_path = os.path.join(local_dir, "training_info.json")
    yaml_path = os.path.join(local_dir, "hparams.yaml")
    collection_name = "NeuralLib: Deep Learning Models for Biosignals Processing"
    create_readme(hparams_file=yaml_path,
                  training_info_file=json_path,
                  output_file=readme_path,
                  collection=collection_name,
                  description=description,
                  model_name=model_name)

    # Create an HfApi instance
    api = HfApi()

    # Check if repository exists
    try:
        api.repo_info(repo_id=repo_name, token=token)
        print(f"Repository '{repo_name}' already exists. Proceeding with upload...")
    except Exception:
        print(f"Repository '{repo_name}' does not exist. Creating it...")
        api.create_repo(repo_id=repo_name, token=token, exist_ok=True)

    # Upload all files
    upload_folder(folder_path=local_dir, repo_id=repo_name, token=token)
    print(f"Model successfully uploaded to Hugging Face: {repo_name}")

    # todo: upload the performance file



