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

是的，**除了 `SearchToolkit` 之外**，CAMEL 官方在其工具生态里已经预封装了很多 **现成可直接调用的 Toolkit 类**，你可以像 `SearchToolkit` 一样实例化，然后用 `.get_tools()` 拿到函数工具列表直接传给 agent。([Camel AI][1])

以下是官方文档列出的 **可立即使用的内置 Toolkit（按功能分类）**：([Camel AI][1])

---

## 一、信息检索 / 研究相关

这些 Toolkit 提供现成的查询、抓取、检索服务：

- `SearchToolkit` — 多引擎 Web 搜索（Google、DuckDuckGo、Wiki 等）([Camel AI][1])
- `ArxivToolkit` — 与 arXiv API 交互，搜索论文并下载摘要/PDF 等([Camel AI][1])
- `GoogleScholarToolkit` — 获取 Google 学术作者和出版物数据([Camel AI][1])
- `PubMedToolkit` — 调用 PubMed e-utilities 获取医学文献数据([Camel AI][1])
- `SemanticScholarToolkit` — 查询 Semantic Scholar 学术数据([Camel AI][1])
- `JinaRerankerToolkit` — 使用 Jina 进行文档 rerank（提升搜索相关性）([Camel AI][1])

---

## 二、数据 / 文件 / 表格处理

这些 Toolkit 能帮助你处理文件和结构化数据：

- `ExcelToolkit` — 提取 Excel/CSV，输出 markdown 表格等([Camel AI][1])
- `OpenAPIToolkit` — 处理 OpenAPI 规范并调用 REST 接口([Camel AI][1])
- `RetrievalToolkit` — 基于向量数据库等进行检索（本地 RAG）([Camel AI][1])

---

## 三、外部服务 / 应用集成

一系列 Toolkit 用于与常见平台的集成：

- `GoogleCalendarToolkit` — 管理日历事件([Camel AI][1])
- `GoogleMapsToolkit` — 地理服务（地址验证、海拔、时区等）([Camel AI][1])
- `SlackToolkit` — Slack 通信与管理([Camel AI][1])
- `LinkedInToolkit` — LinkedIn 帐户操作/帖子创建([Camel AI][1])
- `TwitterToolkit` — Twitter 帖子与用户数据操作([Camel AI][1])
- `StripeToolkit` — 支付与交易处理（Stripe）([Camel AI][1])
- `WhatsAppToolkit` — WhatsApp Business API 交互([Camel AI][1])
- `NotionToolkit` — Notion 页面与工作区访问([Camel AI][1])
- `SlackToolkit`、`RedditToolkit` — 各类社交/通讯平台([Camel AI][1])

---

## 四、数学 / 计算 / 编程

- `MathToolkit` — 基本数学运算（加减乘除等）([Camel AI][1])
- `CodeExecutionToolkit` — 执行代码片段（Python 或 sandbox 环境）([Camel AI][1])
- `SymPyToolkit` — 符号计算（使用 SymPy）([Camel AI][1])
- `TerminalToolkit` — 执行终端/Shell 操作（查文件、run 命令等）([Camel AI][2])

---

## 五、多媒体 / 视觉 / 音频

- `ImageAnalysisToolkit` — 图像分析与理解（依赖视觉模型）([Camel AI][1])
- `VideoAnalysisToolkit` — 视频处理与分析([Camel AI][1])
- `AudioAnalysisToolkit` — 音频转录/分析([Camel AI][1])
- `OpenAIImageToolkit` — 使用 OpenAI DALL-E 等生成图像([Camel AI][1])

---

## 六、工具辅助 / 通用底层

- `MemoryToolkit` — 保存/加载/清除 agent 内存([Camel AI][1])
- `FileWriteToolkit` — 写文件/文本处理([Camel AI][1])
- `MCPToolkit` — 将任意 toolkit 暴露为 MCP server（远程可调用）([Camel AI][1])
- `FunctionTool` — 核心低层工具包装类，用于创建自定义 function tool([Camel AI][1])

---

## 七、如何直接使用这些 Toolkits

你可以按和 `SearchToolkit` 一样的模式来使用它们：

```python
from camel.toolkits import ExcelToolkit, MathToolkit
excel_tools = ExcelToolkit().get_tools()
math_tools  = MathToolkit().get_tools()
tools = [*excel_tools, *math_tools, *SearchToolkit().get_tools()]
agent = ChatAgent(model=model, tools=tools)
```

这样模型就能调用这些工具执行相关任务。([Camel AI][1])

---

## 八、其他注意事项

1. 有些工具依赖额外安装（比如文档处理、浏览器自动化等），需要在安装时加对应 extras，例如：

   ```bash
   pip install 'camel-ai[web_tools]'
   ```

   或更全面的：

   ```bash
   pip install 'camel-ai[all]'
   ```

   这样才能让所有 toolkit 的依赖都可用。([Camel AI][3])

2. 不同 toolkits 所提供的函数，会自动生成规范的 OpenAI schema，模型更容易正确调用。([Camel AI][1])

---

如果你告诉我你希望 agent 做的具体任务（例如“执行代码片段”、“查询天气”、“分析图片”），我可以帮你挑出最适合且**现成可调用**的 Toolkit Class，并给出示例。

[1]: https://docs.camel-ai.org/key_modules/tools?utm_source=chatgpt.com "CAMEL - Build Multi-Agent AI Systems"
[2]: https://docs.camel-ai.org/reference/camel.toolkits.terminal_toolkit?utm_source=chatgpt.com "CAMEL - Build Multi-Agent AI Systems"
[3]: https://docs.camel-ai.org/_modules/camel/toolkits/searxng_toolkit.html?utm_source=chatgpt.com "Installation - CAMEL-AI Documentation"

---

---

---

---

---

---

# 0109 的结果

效果不理想的最大问题：只用大模型的话会幻觉，这是最根源问题，不只是 agent，就连网页版的 gpt 也是一样，
必须要跟他说必须使用工具，然后返回给我 URL 才能最大限度的解决这个问题
所以我认为的解决方案是找到修改的依据，就跟人类程序员一样
但是如果要去 web search 的话很有可能噪声很大，这个效果还没测试
用 RAG 的话可以根源抑制这个问题，这一点要重点和师兄讨论一下

## Web Search Tools 方案

1. 自带的 tools 很难用，最多用 wiki，但是没什么用
2. 先检查 web-search-mcp 这个包好不好用
3. 最好自己写一个完整的 server，比较可控，用爬虫的方案

## RAG 方案还完全没开始

这个路径的难点在于制作文档等收集资料，还是需要爬虫
