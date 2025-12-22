import json

def txt_read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
def jsonl_read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.readlines()]
    
def append_to_jsonl(file_path, item):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')