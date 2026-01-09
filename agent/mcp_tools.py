ANSWER_MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "full-web-search",
            "description": "Perform a full web search for a query and return structured results including full page contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of search results to return (1–10)."
                    },
                    "includeContent": {
                        "type": "boolean",
                        "description": "Whether to fetch full page content for each result."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get-web-search-summaries",
            "description": "Perform a lightweight web search and return titles and snippets only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of summary results (1–10)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get-single-web-page-content",
            "description": "Extract the main content of a single webpage given a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the page to extract content from."
                    },
                    "maxContentLength": {
                        "type": "integer",
                        "description": "Maximum number of characters to return from the page."
                    }
                },
                "required": ["url"]
            }
        }
    }
]

