from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_name="microsoft/DialoGPT-medium", device=-1):
    try:
        print(f"Loading model: {model_name}...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Some GPT models don't have a pad_token; set to eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Create pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        
        print(f"Model loaded successfully!\n")
        return generator
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please check your internet connection and try again.")
        raise

def get_device():
    if torch.cuda.is_available():
        print("GPU detected! Using CUDA.")
        return 0
    else:
        print("No GPU detected. Using CPU.")
        return -1
