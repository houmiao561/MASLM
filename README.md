当前流程：
josnl 单条数据用 AST 预处理，只专注于 具体依赖 以及 API 调用

再将这个预处理结果送到 MAS 中，具体 MAS 中的 Agent 的职责如下：

1. Location ：负责定位哪个 API 调用时候可能出问题了
2. Answer ：负责用自然语言解释这个 API 修改的具体原因
3. Fix ：负责具体修改函数

这样设计的原因是更加贴合 benchmark 的要求

最终结果汇总成一条 JSONL，JSONL 字段如下：
{
原始数据
ast_structure
ai_api_wrong
reason_type
confidence
ai_api_answer_change
ai_api_fix_function
}

`bash` python run.py

以下是 AST 预处理之后保留的字段
"imports": ["numpy"],
"ast_structure":
[{
"function_name": "find_largest_equal_substring",
"lineno": 1,
"api_calls":[
{"api": "range", "lineno": 5, "context": "expression"},
{"api": "len", "lineno": 5, "context": "expression"},
{"api": "range", "lineno": 6, "context": "expression"},
{"api": "len", "lineno": 6, "context": "expression"},
{"api": "len", "lineno": 8, "context": "expression"},
{"api": "len", "lineno": 8, "context": "expression"},
{"api": "numpy.compare_chararrays", "lineno": 9, "context": "expression"}
]
}]
