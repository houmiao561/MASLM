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

| 工具名称                      | 参数                               | 作用                            |
| ----------------------------- | ---------------------------------- | ------------------------------- |
| `full-web-search`             | `query`, `limit`, `includeContent` | 全网页搜索 + 可抓取全部页面内容 |
| `get-web-search-summaries`    | `query`, `limit`                   | 只获取搜索结果的简要摘要        |
| `get-single-web-page-content` | `url`, `maxContentLength`          | 单页面内容提取                  |

<!-- camel自带 -->

| 方法                                                                    | 功能                               |
| ----------------------------------------------------------------------- | ---------------------------------- |
| `search_google(query, search_type, number_of_result_pages, start_page)` | 使用 Google 结果进行搜索           |
| `search_wiki(entity)`                                                   | 在维基百科搜索实体并返回简要摘要   |
| `search_serper(query)`                                                  | 通过 Serper API 做 Google 样式搜索 |
| `search_linkup(query, ...)`                                             | 通过 Linkup 接口做结构化搜索       |
| 其他搜索方法（如 brave、duckduckgo 等）                                 | 支持不同搜索源                     |

{'type': 'function', 'function': {'name': 'search_alibaba_tongxiao', 'description': 'Query the Alibaba Tongxiao search API and return search results.\nA powerful search API optimized for Chinese language queries with\nfeatures:\n- Enhanced Chinese language understanding\n- Industry-specific filtering (finance, law, medical, etc.)\n- Structured data with markdown formatting\n- Result reranking for relevance\n- Time-based filtering', 'strict': True, 'parameters': {'properties': {'query': {'type': 'string', 'description': 'The search query string (length >= 1 and <= 100).'}, 'time_range': {'enum': ['OneDay', 'OneWeek', 'OneMonth', 'OneYear', 'NoLimit'], 'type': 'string'}, 'industry': {'anyOf': [{'enum': ['finance', 'law', 'medical', 'internet', 'tax', 'news_province', 'news_center'], 'type': 'string'}, {'type': 'null'}]}, 'page': {'type': 'integer', 'description': 'Page number for results pagination.\n(default: :obj:`1`)'}, 'return_main_text': {'type': 'boolean', 'description': 'Whether to include the main text of the\nwebpage in results. (default: :obj:`True`)'}, 'return_markdown_text': {'type': 'boolean', 'description': 'Whether to include markdown formatted\ncontent in results. (default: :obj:`True`)'}, 'enable_rerank': {'type': 'boolean', 'description': 'Whether to enable result reranking. If\nresponse time is critical, setting this to False can reduce\nresponse time by approximately 140ms. (default: :obj:`True`)'}}, 'required': ['query', 'time_range', 'industry', 'page', 'return_main_text', 'return_markdown_text', 'enable_rerank'], 'type': 'object', 'additionalProperties': False}}}
