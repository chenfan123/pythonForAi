import warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer

# # define the sentence to tokenize
# sentence = "Hello world!"

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") # 分词器，使用预训练好的 BERT 模型

# tokens = tokenizer.tokenize(sentence)

# # print the tokens
# print(tokens) # ['hello', 'world', '!']

# # print the token ids
# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(token_ids)  # [8667, 1362, 106]

# for token_id in token_ids:
#     print(tokenizer.decode([token_id]))  # Hello world !

# 颜色列表
colors = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]
def show_tokens(sentence: str, tokenizer_name: str):
    """ Show the tokens each separated by a different color """

    # Load the tokenizer and tokenize the input
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Extract vocabulary length
    print(f"Vocab length: {len(tokenizer)}")

    # Print a colored list of tokens
    for idx, t in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors[idx % len(colors)]}m' +
            tokenizer.decode([t]) +
            '\x1b[0m',
            end=' '
        )

text = """
English and CAPITALIZATION
🎵 鸟
show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "
12.0*50=600
"""

# show_tokens(text, "bert-base-cased")  # 28996长度
# show_tokens(text, "bert-base-uncased")  # 30522长度
show_tokens(text, "Xenova/gpt-4") # 100263长度