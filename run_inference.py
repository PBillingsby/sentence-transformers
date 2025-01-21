import os
import json
import sys
import traceback
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(input_text, model, tokenizer, candidate_texts, max_length=128):
    """
    Compute cosine similarity between the input text and a list of candidate texts.
    """
    try:
        # Tokenize and encode the input text
        input_emb = model(
            **tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
        ).last_hidden_state.mean(dim=1)

        # Tokenize and encode candidate texts
        candidate_embs = model(
            **tokenizer(candidate_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        ).last_hidden_state.mean(dim=1)

        # Compute cosine similarity
        similarities = cosine_similarity(input_emb.detach().numpy(), candidate_embs.detach().numpy()).flatten()

        # Convert similarities to Python float for JSON serialization
        return [float(sim) for sim in similarities]

    except Exception as e:
        print(f"Error computing similarity: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise


def main():
    print("Starting text similarity computation", file=sys.stderr, flush=True)

    # Get environment variables
    input_text = os.environ.get('INPUT_TEXT', 'Default input text')
    candidate_texts = os.environ.get('CANDIDATE_TEXTS', '["Text A", "Text B", "Text C"]')

    output = {
        'input_text': input_text,
        'status': 'error',
        'similarities': {}
    }

    try:
        # Parse candidate texts
        candidate_texts = json.loads(candidate_texts)
        if not isinstance(candidate_texts, list):
            raise ValueError("CANDIDATE_TEXTS must be a JSON-encoded list of strings.")

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained('/model')
        model = AutoModel.from_pretrained('/model')

        # Compute similarities
        similarities = compute_similarity(input_text, model, tokenizer, candidate_texts)
        output.update({
            'status': 'success',
            'similarities': {text: sim for text, sim in zip(candidate_texts, similarities)}
        })

        for text, sim in output['similarities'].items():
            print(f"{text}: {sim:.4f}", file=sys.stderr, flush=True)

    except Exception as e:
        print("Error during processing:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        output['error'] = str(e)

    # Save output to file
    os.makedirs('/outputs', exist_ok=True)
    output_path = '/outputs/result.json'

    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Successfully wrote output to {output_path}", file=sys.stderr, flush=True)
    except Exception as write_error:
        print("Error writing output file:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
