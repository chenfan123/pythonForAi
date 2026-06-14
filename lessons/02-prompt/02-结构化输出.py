from openai_lesson import get_completion    
prompt = f"""
生成一个包含 3 本虚构书籍的列表，
并附上每本书的作者和类型。
请使用 JSON 格式输出，并包含以下键：
book_id、title、author、genre。
"""

response = get_completion(prompt)
print(response)