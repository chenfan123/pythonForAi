from pathlib import Path
import importlib.util

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings
warnings.filterwarnings('ignore')
# import the required classes

# Prefer local model directory; fallback to Hugging Face repo id.
LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "microsoft"
    / "Phi-3-mini-4k-instruct"
)
MODEL_ID = (
    str(LOCAL_MODEL_DIR)
    if LOCAL_MODEL_DIR.exists()
    else "microsoft/Phi-3-mini-4k-instruct"
)

if importlib.util.find_spec("torch") is None:
    raise RuntimeError(
        "未检测到 PyTorch。请先安装 torch，例如：pip install torch"
    )

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)

# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # False means to not include the prompt text in the returned text
    return_full_text=False,
    max_new_tokens=50,
    do_sample=False,  # no randomness in the generated text
)

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened. "

output = generator(prompt)

print(output[0]['generated_text'])
