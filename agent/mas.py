import json
import re
import sys
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents.chat_agent import ChatAgent
from utils import *

def create_agent(system_prompt: str):
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type="deepseek-ai/DeepSeek-V3",
        model_config_dict={
            "temperature": 0.7,
            "max_tokens": 4096,
        },
    )

    agent = ChatAgent(
        system_message=system_prompt,
        model=model,
    )
    return agent

class Orchestrator:
    def __init__(self, agents: dict, raw_data: dict):
        self.agents = agents
        self.raw_data = raw_data
        self.token_stats = {
            "location_library": 0,
            "answer_change": 0,
            "fix_function": 0,
            "total": 0,
        }
    
    def _extract_tokens(self, response):
        """
        Best-effort extraction of token usage from CAMEL response.
        Compatible with multiple CAMEL / backend versions.
        """
        usage = None

        if hasattr(response, "usage") and response.usage:
            usage = response.usage
        elif hasattr(response, "info") and response.info and "usage" in response.info:
            usage = response.info["usage"]
        elif hasattr(response, "model_response") and response.model_response:
            usage = response.model_response.get("usage")

        if not usage:
            return 0

        return (
            usage.get("total_tokens", 0)
            or usage.get("total", 0)
            or (
                usage.get("prompt_tokens", 0)
                + usage.get("completion_tokens", 0)
            )
        )

    def _extract_json(self, text: str):
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                raise ValueError("No JSON found:\n" + text)
            return json.loads(match.group())

    def _get_content(self, response):
        # 兼容不同 CAMEL 小版本
        if hasattr(response, "msg"):
            return response.msg.content
        if hasattr(response, "content"):
            return response.content
        return str(response)
    
    def _location_get_variables(self):
        print("location_get_variables")
        return {
            "compare_version":self.raw_data['compare_version'],
            "package":self.raw_data['package'],
            "solution_function":self.raw_data['solution_function'],
            "ast_structure":self.raw_data['ast_structure']
        }
    
    def _answer_get_variables(self):
        print("answer_get_variables")
        return {
            "compare_version":self.raw_data['compare_version'],
            "package":self.raw_data['package'],
            "solution_function":self.raw_data['solution_function'],
            "ast_structure":self.raw_data['ast_structure'],
            "ai_api_wrong":self.raw_data['ai_api_wrong'],
            "line_number":self.raw_data['line_number']   
        }
    
    def _fix_function_get_variables(self):
        print("fix_function_get_variables")
        return {
            "compare_version":self.raw_data['compare_version'],
            "package":self.raw_data['package'],
            "solution_function":self.raw_data['solution_function'],
            "ast_structure":self.raw_data['ast_structure'],
            "ai_api_wrong":self.raw_data['ai_api_wrong'],
            "line_number":self.raw_data['line_number'],
            "reason_type":self.raw_data['reason_type'],
            "ai_api_answer_change":self.raw_data['ai_api_answer_change']
        }

    def location_library(self):
        print("location_library")
        # 这里的 task 就是 prompt 中的==========User Input==========部分的输入，而且task是字符串类型，因为LLM只能读这个
        task = self._location_get_variables()
        resp = self.agents["location_library"].step(str(task)) # str(task)

        tokens = self._extract_tokens(resp)
        self.token_stats["location_library"] += tokens
        self.token_stats["total"] += tokens

        content = self._get_content(resp) # content 是 str 类型
        result = self._extract_json(content)
        self.raw_data["ai_api_wrong"] = result["ai_api_wrong"]
        self.raw_data["line_number"] = result["line_number"]
        return self.raw_data
    
    def answer_change(self):
        print("answer_change")
        task = self._answer_get_variables()
        resp = self.agents["answer_change"].step(str(task)) # str(task)

        tokens = self._extract_tokens(resp)
        self.token_stats["answer_change"] += tokens
        self.token_stats["total"] += tokens

        content = self._get_content(resp) 
        result = self._extract_json(content)
        self.raw_data["ai_api_answer_change"] = result["ai_api_answer_change"] 
        self.raw_data["reason_type"] = result["reason_type"]
        return self.raw_data
        
    def fix_function(self):
        print("fix_function")
        task = self._fix_function_get_variables()
        # task = json.dumps(self.raw_data, ensure_ascii=False)
        resp = self.agents["fix_function"].step(str(task)) # str(task)

        tokens = self._extract_tokens(resp)
        self.token_stats["fix_function"] += tokens
        self.token_stats["total"] += tokens

        content = self._get_content(resp) 
        result = self._extract_json(content)
        self.raw_data["ai_api_fix_function"] = result["ai_api_fix_function"]
        return self.raw_data