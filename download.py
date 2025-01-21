from transformers import AutoTokenizer, AutoModel

def download_and_save_model(model_name: str, save_path: str = "./model"):
    """
    Downloads a transformer model and tokenizer, and saves them locally.

    Args:
        model_name (str): The name of the model to download.
        save_path (str): The directory where the model and tokenizer will be saved.
    """
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Save tokenizer and model to the specified directory
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)

        print(f"Model and tokenizer saved to '{save_path}'")
    except Exception as e:
        print(f"An error occurred while downloading or saving the model: {e}")

if __name__ == "__main__":
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    SAVE_PATH = "./model"
    download_and_save_model(MODEL_NAME, SAVE_PATH)
