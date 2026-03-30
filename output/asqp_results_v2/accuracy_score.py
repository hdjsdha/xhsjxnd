import json

# 定义要读取的JSON文件及其对应的LLM类型
json_files = {
    'llama3b': 'llama3b.json',
    'llama8b': 'llama8b.json',
    'llama70b': 'llama70b.json',
    'llama405b': 'llama405b.json',
    'deepseek': 'deepseek.json',
    'gpt3.5': 'gpt3.5.json',
    'gpt4o': 'gpt4o.json',
    # 可以在这里添加其他的json文件
}

# # 从每个JSON文件中加载数据
# data = {}
# for llm_type, file_name in json_files.items():
#     with open(file_name, 'r', encoding='utf-8') as f:
#         data[llm_type] = json.load(f)

# # 提取前100个text并汇总
# aggregated_results = []

# for i in range(1867):
#     text = data[next(iter(data))][i]['text']
#     sentiment = data[next(iter(data))][i]['overall_sentiment']
#     result_entry = {'text': text, 'overall_sentiment': sentiment}
#     for llm_type in data:
#         result = data[llm_type][i]['sentiment_quadruples']
#         result_entry[llm_type] = result
#     aggregated_results.append(result_entry)

# with open('aggregated_asqp_results.json', 'w', encoding='utf-8') as f:
#     json.dump(aggregated_results, f, ensure_ascii=False, indent=4)

# Load the verified data
with open('llama405b_deepseek_verified.json', 'r', encoding='utf-8') as f:
    verified_data = json.load(f)

# Collect entries where 'is_valid' is False
invalid_entries = []

for entry in verified_data:
    verification_results = entry.get('verification_results', {})
    validations = verification_results.get('validations', [])
    is_valid = verification_results.get('is_valid', True)
    if not is_valid:
        invalid_entry = {
            'text': entry.get('text', ''),
            'overall_sentiment': entry.get('overall_sentiment', ''),
            'final_quadruples': entry.get('final_quadruples', [])
        }
        invalid_entries.append(invalid_entry)

# Write the invalid entries to a new JSON file
with open('invalid_entries.json', 'w', encoding='utf-8') as f:
    json.dump(invalid_entries, f, ensure_ascii=False, indent=4)
