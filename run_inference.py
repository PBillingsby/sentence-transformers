import os
import json
import sys
import logging
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class SimilarityError(Exception):
    """Custom exception for similarity computation errors"""
    pass

def setup_logging() -> None:
    """Configure logging settings"""
    # Ensure all handlers are removed to avoid duplicate logging
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add stderr handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def load_environment_variables() -> Tuple[str, str]:
    """
    Load and validate environment variables
    Returns: Tuple of (input_text, candidate_texts_raw)
    """
    logger.info("Loading environment variables")
    
    # Log all environment variables for debugging
    logger.debug("Environment variables:")
    for key, value in sorted(os.environ.items()):
        if key in ['INPUT_TEXT', 'CANDIDATE_TEXTS']:
            logger.debug(f"{key}: {repr(value)}")
    
    input_text = os.environ.get('INPUT_TEXT')
    candidate_texts_raw = os.environ.get('CANDIDATE_TEXTS')
    
    if input_text is None:
        logger.error("INPUT_TEXT environment variable is not set")
        raise ValueError("INPUT_TEXT environment variable is required")
        
    if candidate_texts_raw is None:
        logger.error("CANDIDATE_TEXTS environment variable is not set")
        raise ValueError("CANDIDATE_TEXTS environment variable is required")
    
    logger.info(f"Loaded INPUT_TEXT: {repr(input_text)}")
    logger.info(f"Loaded CANDIDATE_TEXTS: {repr(candidate_texts_raw)}")
    
    return input_text, candidate_texts_raw

def parse_candidate_texts(candidate_texts_raw: str) -> List[str]:
    """
    Parse and validate candidate texts from JSON string
    Args:
        candidate_texts_raw: JSON string of candidate texts
    Returns:
        List of candidate text strings
    """
    logger.info("Parsing candidate texts")
    logger.debug(f"Raw candidate texts: {repr(candidate_texts_raw)}")
    
    try:
        candidate_texts = json.loads(candidate_texts_raw)
        
        if not isinstance(candidate_texts, list):
            logger.error(f"Expected list type, got {type(candidate_texts)}")
            raise ValueError("CANDIDATE_TEXTS must be a JSON-encoded list of strings")
            
        if not all(isinstance(text, str) for text in candidate_texts):
            logger.error("Not all elements are strings")
            raise ValueError("All elements in CANDIDATE_TEXTS must be strings")
        
        logger.info(f"Successfully parsed {len(candidate_texts)} candidate texts")
        logger.debug(f"Parsed texts: {candidate_texts}")
        return candidate_texts
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {str(e)}")
        logger.error(f"Error position: char {e.pos}")
        raise ValueError(f"Failed to parse CANDIDATE_TEXTS as JSON: {str(e)}")

def load_model() -> Tuple[Any, Any]:
    """
    Load the transformer model and tokenizer
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Loading model and tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained('/model')
        model = AutoModel.from_pretrained('/model')
        logger.info("Successfully loaded model and tokenizer")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {str(e)}")
        raise SimilarityError(f"Model loading failed: {str(e)}")

def compute_similarity(
    input_text: str,
    model: Any,
    tokenizer: Any,
    candidate_texts: List[str],
    max_length: int = 128
) -> Dict[str, float]:
    """
    Compute cosine similarity between input text and candidate texts
    Returns:
        Dictionary mapping candidate texts to their similarity scores
    """
    logger.info(f"Computing similarities for input: '{input_text}'")
    logger.info(f"Number of candidate texts: {len(candidate_texts)}")
    
    try:
        # Tokenize and encode input text
        input_encoding = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        logger.debug(f"Input text encoded shape: {input_encoding.input_ids.shape}")
        
        input_emb = model(**input_encoding).last_hidden_state.mean(dim=1)
        
        # Tokenize and encode candidate texts
        candidate_encoding = tokenizer(
            candidate_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        logger.debug(f"Candidate texts encoded shape: {candidate_encoding.input_ids.shape}")
        
        candidate_embs = model(**candidate_encoding).last_hidden_state.mean(dim=1)
        
        # Compute similarities
        similarities = cosine_similarity(
            input_emb.detach().numpy(),
            candidate_embs.detach().numpy()
        ).flatten()
        
        # Create results dictionary
        results = {text: float(sim) for text, sim in zip(candidate_texts, similarities)}
        logger.info("Successfully computed similarities")
        logger.debug(f"Similarity scores: {results}")
        
        return results

    except Exception as e:
        logger.error(f"Error computing similarities: {str(e)}")
        raise SimilarityError(f"Similarity computation failed: {str(e)}")

def save_output(output: Dict[str, Any], output_path: str = '/outputs/result.json') -> None:
    """
    Save output dictionary to JSON file
    Args:
        output: Dictionary to save
        output_path: Path to save JSON file
    """
    logger.info(f"Saving output to {output_path}")
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info("Successfully saved output")
        
    except Exception as e:
        logger.error(f"Failed to save output: {str(e)}")
        # Print to stderr as fallback
        print(json.dumps(output, indent=2), file=sys.stderr)
        raise

def main() -> None:
    """Main function to run the similarity computation pipeline"""
    setup_logging()
    logger.info("Starting text similarity computation")
    
    output = {
        'input_text': None,
        'status': 'error',
        'similarities': {}
    }
    
    try:
        # Load inputs
        input_text, candidate_texts_raw = load_environment_variables()
        output['input_text'] = input_text
        
        # Parse candidate texts
        candidate_texts = parse_candidate_texts(candidate_texts_raw)
        
        # Load model and compute similarities
        model, tokenizer = load_model()
        similarities = compute_similarity(input_text, model, tokenizer, candidate_texts)
        
        # Update output
        output.update({
            'status': 'success',
            'similarities': similarities
        })
        
        # Log results
        for text, score in similarities.items():
            logger.info(f"Similarity for '{text}': {score:.4f}")
            
    except Exception as e:
        logger.error("Error in main processing", exc_info=True)
        output['error'] = str(e)
    
    # Always try to save output
    try:
        save_output(output)
    except Exception as e:
        logger.error(f"Could not save output file: {str(e)}")

if __name__ == "__main__":
    main()