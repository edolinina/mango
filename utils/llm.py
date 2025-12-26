from langchain_ollama import ChatOllama

# Ollama model configuration
OLLAMA_BASE_URL = "http://192.168.1.120:11434"
OLLAMA_MODEL_NAME = "gpt-oss:20b" # llm model
OLLAMA_MODEL_TEMPERATURE = 0.2 # for more deterministic responses
OLLAMA_MODEL_SEED = 42 # for repetitive responses
OLLAMA_MODEL_GPUS = 1
OLLAMA_MODEL_THREADS = 20

# --------------------
# MODEL LOADING
# --------------------
def load_model() -> ChatOllama:
    """
    Load the Ollama LLM model.

    Returns:
        ChatOllama: Configured Ollama model instance.
    """
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL_NAME,
        temperature=OLLAMA_MODEL_TEMPERATURE
    )