from python_for_ai import get_llm_response, print_llm_response

print(round(1.23456789, 2))  # 1.23 四舍五入

# get_llm_response：返回字符串，便于你自己处理
response = get_llm_response("Hello, how are you?")
print(response)

# print_llm_response：直接打印带边框的回复
print_llm_response("Hello, how are you?")
