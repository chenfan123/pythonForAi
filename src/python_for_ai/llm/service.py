from python_for_ai.config import MODEL_NAME
from python_for_ai.llm.client import get_client


def get_llm_response(prompt: str) -> str:
    """Call the LLM and return the model response as a string."""
    if not isinstance(prompt, str):
        raise ValueError("Input must be a string enclosed in quotes.")

    completion = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful but terse AI assistant "
                    "who gets straight to the point."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    content = completion.choices[0].message.content
    if content is None:
        raise ValueError("Model returned empty content.")
    return content


def print_llm_response(prompt: str) -> None:
    """Call the LLM and print the model response."""
    try:
        response = get_llm_response(prompt)
        print("*" * 100)
        print(response)
        print("*" * 100)
        print("\n")
    except TypeError as error:
        print("Error:", str(error))
