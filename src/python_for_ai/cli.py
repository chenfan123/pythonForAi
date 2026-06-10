import sys

from python_for_ai.llm.service import print_llm_response


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python-for-ai <prompt>")
        sys.exit(1)
    print_llm_response(sys.argv[1])


if __name__ == "__main__":
    main()
