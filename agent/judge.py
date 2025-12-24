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
            "max_tokens": 2048,
        },
    )

    agent = ChatAgent(
        system_message=system_prompt,
        model=model,
    )
    return agent

class OrchestratorJudger:
    def __init__(self, agents: dict, raw_data: dict):
        self.agents = agents
        self.raw_data = raw_data
        self.token_stats = {
            "judger": 0,
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
    
    def _judger_get_variables(self):
        print("_judger_get_variables")
        return {
            "solution_function":self.raw_data['solution_function'],
            "signature":self.raw_data['signature'],
            "update":self.raw_data['update'],
            "ai_api_wrong":self.raw_data['ai_api_wrong'],
            "ai_api_answer_change":self.raw_data['ai_api_answer_change']
        }

    def judgeFunction(self):
        print("judgeFunction")
        # 这里的 task 就是 prompt 中的==========User Input==========部分的输入，而且task是字符串类型，因为LLM只能读这个
        task = self._judger_get_variables()
        resp = self.agents["judger"].step(str(task)) # str(task)

        tokens = self._extract_tokens(resp)
        self.token_stats["judger"] += tokens
        self.token_stats["total"] += tokens

        content = self._get_content(resp) # content 是 str 类型
        result = self._extract_json(content)
        self.raw_data["judge_reason"] = result["judge_reason"]
        self.raw_data["judge_locate_answer"] = result["judge_locate_answer"]
        self.raw_data["judge_update_answer"] = result["judge_update_answer"]
        return self.raw_data