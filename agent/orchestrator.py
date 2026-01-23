import json,re,sys

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
            "line_number":self.raw_data['line_number'],
            "natural_language_questions":self.raw_data['natural_language_questions']  
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
            "natural_language_questions":self.raw_data['natural_language_questions'],
            "reason_type":self.raw_data['reason_type'],
            "ai_api_answer_change":self.raw_data['ai_api_answer_change'],
            "mcp_evidence_summary":self.raw_data['mcp_evidence_summary']
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
        self.raw_data["natural_language_questions"] = result["natural_language_questions"]
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
        self.raw_data["mcp_evidence_summary"] = result["mcp_evidence_summary"]
        return self.raw_data
        
    def fix_function(self):
        print("fix_function")
        task = self._fix_function_get_variables()
        resp = self.agents["fix_function"].step(str(task)) # str(task)

        tokens = self._extract_tokens(resp)
        self.token_stats["fix_function"] += tokens
        self.token_stats["total"] += tokens

        content = self._get_content(resp) 
        result = self._extract_json(content)
        self.raw_data["ai_api_fix_function"] = result["ai_api_fix_function"]
        return self.raw_data
    

class OrchestratorHardPy:
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
        print("location_get_variables_hardpy")
        self.raw_data['ai_api_fix_function'] = self.raw_data['solution_function']
        return {            
            "compare_version":self.raw_data['compare_version'],
            "package":self.raw_data['package'],
            "solution_function":self.raw_data['solution_function'],
            "ast_structure":self.raw_data['ast_structure']
        }
    
    def _answer_get_variables(self):
        print("answer_get_variables_hardpy")
        return {
            "compare_version":self.raw_data['compare_version'],
            "package":self.raw_data['package'],
            "solution_function":self.raw_data['solution_function'],
            "ast_structure":self.raw_data['ast_structure']
        }
    
    def _fix_function_get_variables(self):
        print("fix_function_get_variables_hardpy")
        return {
            "compare_version":self.raw_data['compare_version'],
            "package":self.raw_data['package'],
            "ast_structure":self.raw_data['ast_structure'],
            "ai_api_fix_function":self.raw_data['ai_api_fix_function']
        }

    def location_library(self):
        print("location_library_hardpy")
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
        self.raw_data["natural_language_questions"] = result["natural_language_questions"]
        return self.raw_data
    
    def answer_change(self, single_api_index):
        print("answer_change_hardpy")
        task = self._answer_get_variables()
        task["single_api_index"] = single_api_index
        task["ai_api_wrong"] = self.raw_data["ai_api_wrong"][single_api_index]
        task["line_number"] = self.raw_data["line_number"][single_api_index]
        task["natural_language_questions"] = self.raw_data["natural_language_questions"][single_api_index]
        resp = self.agents["answer_change"].step(str(task)) # str(task)
        tokens = self._extract_tokens(resp)
        self.token_stats["answer_change"] += tokens
        self.token_stats["total"] += tokens


        content = self._get_content(resp) 
        result = self._extract_json(content)

        self.raw_data.setdefault("ai_api_answer_change", []).append(result["ai_api_answer_change"])
        self.raw_data.setdefault("reason_type", []).append(result["reason_type"])
        self.raw_data.setdefault("mcp_evidence_summary", []).append(result["mcp_evidence_summary"])
        return self.raw_data
        
    def fix_function(self, single_api_index):
        print("fix_function_hardpy")
        task = self._fix_function_get_variables()
        task["single_api_index"] = single_api_index
        task["ai_api_wrong"] = self.raw_data["ai_api_wrong"][single_api_index]
        task["line_number"] = self.raw_data["line_number"][single_api_index]
        task["natural_language_questions"] = self.raw_data["natural_language_questions"][single_api_index]
        task["reason_type"] = self.raw_data["reason_type"][single_api_index]
        task["ai_api_answer_change"] = self.raw_data["ai_api_answer_change"][single_api_index]
        task["mcp_evidence_summary"] = self.raw_data["mcp_evidence_summary"][single_api_index]
        
        resp = self.agents["fix_function"].step(str(task)) # str(task)

        tokens = self._extract_tokens(resp)
        self.token_stats["fix_function"] += tokens
        self.token_stats["total"] += tokens

        content = self._get_content(resp) 
        result = self._extract_json(content)
        self.raw_data["ai_api_fix_function"] = result["ai_api_fix_function"] # 多轮循环直接覆盖掉
        return self.raw_data
    

class OrchestratorJava:
    def __init__(self, agents: dict, raw_data: dict):
        self.agents = agents
        self.raw_data = raw_data
        self.token_stats = {
            "location_library": 0,
            "answer_change": 0,
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
            "java_code":self.raw_data['java_code'],
            "version":"JDK11",
            "ast_structure":self.raw_data["ast_structure"],
        }
    
    def _answer_get_variables(self):
        print("answer_get_variables")
        return {
            "java_code":self.raw_data['java_code'],
            "version":"JDK11",
            "ast_structure":self.raw_data["ast_structure"],
            "ai_api_wrong":self.raw_data["ai_api_wrong"],
            "ai_api_location":self.raw_data["ai_api_location"],
        }
    
    def location_library(self):
        print("location_library_java")
        task = self._location_get_variables()
        
        resp = self.agents["location_library"].step(str(task)) # str(task)
        tokens = self._extract_tokens(resp)
        self.token_stats["location_library"] += tokens
        self.token_stats["total"] += tokens

        content = self._get_content(resp) 
        result = self._extract_json(content)

        self.raw_data["ai_api_wrong"] = result["ai_api_wrong"]
        self.raw_data["ai_api_location"] = result["ai_api_location"]
        return self.raw_data
    
    def answer_change(self):
        print("answer_change_java")
        task = self._answer_get_variables()
        
        resp = self.agents["answer_change"].step(str(task)) # str(task)
        tokens = self._extract_tokens(resp)
        self.token_stats["answer_change"] += tokens
        self.token_stats["total"] += tokens

        content = self._get_content(resp) 
        result = self._extract_json(content)

        self.raw_data["ai_api_answer_change"] = result["ai_api_answer_change"] 
        self.raw_data["reason_type"] = result["reason_type"]
        self.raw_data["mcp_evidence_summary"] = result["mcp_evidence_summary"]
        return self.raw_data
