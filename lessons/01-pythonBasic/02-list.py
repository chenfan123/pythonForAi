from python_for_ai import get_llm_response, print_llm_response

friends_list = ["张三", "李四", "王五", "赵六", "孙七"]
print(friends_list)

friends_list.append("周八")
print(friends_list)

friends_list.insert(0, "陈九")
print(friends_list)

friends_list.remove("李四")
print(friends_list)

friends_list.pop()
print(friends_list)

friends_list.clear()
print(friends_list)

list_of_task = [
'你好，明天天气怎么样？',
'明天吃什么？',
'明天去哪里玩？',
]

results_from_llm = []
for task in list_of_task:
    results_from_llm.append(get_llm_response(task))

print(results_from_llm)


