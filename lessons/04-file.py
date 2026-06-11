from pathlib import Path

from python_for_ai import get_llm_response, print_llm_response
from IPython.display import display, Markdown

file_path = Path(__file__).resolve().parent / "04-file.txt"

my_file_content = ""
with open(file_path, "r", encoding="utf-8") as file:
    my_file_content = file.read()
    file.close()

print_llm_response(my_file_content)
