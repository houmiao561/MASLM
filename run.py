import os
import sys
import time
from ast_pre.ast_pre import extract_ast_structure
from ast_pre.ast_pre_java import extract_java_ast_structure
from utils import *
from agent.orchestrator import Orchestrator,OrchestratorHardPy,OrchestratorJava
from agent.create_single_agent import create_agent
from agent.judger import OrchestratorJudger

def maslm():
    LOCATION_AGENT_PROMPT = txt_read_file("prompt/easy_python/location.txt")
    ANSWER_CHANGE_AGENT_PROMPT = txt_read_file("prompt/easy_python/answer.txt")
    FIX_FUNCTION_AGENT_PROMPT = txt_read_file("prompt/easy_python/fix_function.txt")
    FINAL_TOKEN = 0
    data = jsonl_read_file("input_dataset/test_case.jsonl")
    data = data[:50]
    for index, CODE in enumerate(data):
        print(f"Processing COOOOOODE {index}...")
        # AST 预处理
        ast_result = extract_ast_structure(CODE)
        sample = {**CODE, **ast_result} # 拼接

        # MAS启动
        agents = {
            "location_library": create_agent(LOCATION_AGENT_PROMPT), 
            "answer_change": create_agent(ANSWER_CHANGE_AGENT_PROMPT,server_url= "https://mcp.context7.com/mcp",api_key="ctx7sk-97bd7e64-9cb4-477e-a13e-51c267f58e6e"),
            "fix_function": create_agent(FIX_FUNCTION_AGENT_PROMPT)
        }

        orch = Orchestrator(agents, sample)


        # MAS具体执行三个Agent,前两个不执行没法执行第三个
        location_result = orch.location_library()
        answer_change_result = orch.answer_change()
        fix_function_result = orch.fix_function()


        # 结果写入并print
        append_to_jsonl("output_dataset/easy_python/create_result.jsonl", fix_function_result)
        print("\n========== TOKEN USAGE SUMMARY ==========")
        for k, v in orch.token_stats.items():
            print(f"{k}: {v}")
        print("========================================\n")
        FINAL_TOKEN += orch.token_stats["total"]

    print("\n========== TOKEN USAGE SUMMARY ==========")
    print(f"FINAL_TOKEN: {FINAL_TOKEN}")
    print("========================================\n")

def judge_bench():
    JUDGE_AGENT_PROMPT = txt_read_file("prompt/easy_python/judger.txt")
    FINAL_TOKEN = 0
    data = jsonl_read_file("/Users/houmiao/Desktop/MASLM/output_dataset/easy_python/create_result.jsonl")
    print(len(data))
    # sys.exit()
    # 选择前50个
    data = data[:50]
    for index, CODE in enumerate(data):
        print(f"JUDGE COOOOOODE {index}...")
        # MAS启动
        agents = {
            "judger": create_agent(JUDGE_AGENT_PROMPT), 
        }
        orch = OrchestratorJudger(agents, CODE)
        # MAS具体执行三个Agent
        
        judge_result = orch.judgeFunction()
        print(judge_result)

        # 结果写入并print
        append_to_jsonl("output_dataset/easy_python/judge_result_change_prompt.jsonl", judge_result)
        print("\n========== TOKEN USAGE SUMMARY ==========")
        for k, v in orch.token_stats.items():
            print(f"{k}: {v}")
        print("========================================\n")
        FINAL_TOKEN += orch.token_stats["total"]

    print("\n========== TOKEN USAGE SUMMARY ==========")
    print(f"FINAL_TOKEN: {FINAL_TOKEN}")
    print("========================================\n")

def compute_avg(jsonl_path):
    # data = jsonl_read_file("output_dataset/easy_python/judge_result.jsonl")
    # data = jsonl_read_file(jsonl_path)
    total = 0
    locate_sum = 0
    update_sum = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            locate = int(data.get("judge_locate_answer", 0))
            update = int(data.get("judge_update_answer", 0))

            locate_sum += locate
            update_sum += update
            total += 1

    if total == 0:
        raise ValueError("Empty jsonl file")

    task1_avg = locate_sum / total
    task2_avg = update_sum / total

    return {
        "total_samples": total,
        "task1_avg": round(task1_avg * 100, 2),
        "task2_avg": round(task2_avg * 100, 2),
    }

def maslm_hard_python():
    LOCATION_AGENT_PROMPT = txt_read_file("prompt/hard_python/location.txt")
    ANSWER_CHANGE_AGENT_PROMPT = txt_read_file("prompt/hard_python/answer.txt")
    FIX_FUNCTION_AGENT_PROMPT = txt_read_file("prompt/hard_python/fix_function.txt")
    FINAL_TOKEN = 0
    data = jsonl_read_file("input_dataset/test_case.jsonl")
    for index, CODE in enumerate(data):
        print(f"Processing COOOOOODE {index}...")
        # AST 预处理
        ast_result = extract_ast_structure(CODE)
        sample = {**CODE, **ast_result} # 拼接

        # MAS启动
        agents = {
            "location_library": create_agent(LOCATION_AGENT_PROMPT), 
            "answer_change": create_agent(ANSWER_CHANGE_AGENT_PROMPT,server_url= "https://mcp.context7.com/mcp",api_key="ctx7sk-97bd7e64-9cb4-477e-a13e-51c267f58e6e"),
            "fix_function": create_agent(FIX_FUNCTION_AGENT_PROMPT)
        }

        orch = OrchestratorHardPy(agents, sample)

        # Location
        location_result = orch.location_library()
        
        # # Answer
        for single_api_index, single_api_name in enumerate(location_result["ai_api_wrong"]):
            print(single_api_index, single_api_name)
            answer_change_result = orch.answer_change(single_api_index)
            print(orch.raw_data["ai_api_answer_change"])
            print(orch.raw_data["reason_type"])
            print(orch.raw_data["mcp_evidence_summary"])
            orch.agents["answer_change"].clear_memory()
            print()
            print()
        
        # Fix
        for single_api_index, single_api_name in enumerate(orch.raw_data["ai_api_wrong"]):
            print(single_api_index, single_api_name)
            fix_function_result = orch.fix_function(single_api_index)
            print(orch.raw_data["ai_api_fix_function"])
            orch.agents["fix_function"].clear_memory()
            print()
            print()
        
        print(f"FINALRESULTTTT:::::\n{orch.raw_data}")

        # 结果写入并print
        append_to_jsonl("output_dataset/hard_python/create_result.jsonl", fix_function_result)
        print("\n========== TOKEN USAGE SUMMARY ==========")
        for k, v in orch.token_stats.items():
            print(f"{k}: {v}")
        print("========================================\n")
        FINAL_TOKEN += orch.token_stats["total"]

    print("\n========== TOKEN USAGE SUMMARY ==========")
    print(f"FINAL_TOKEN: {FINAL_TOKEN}")
    print("========================================\n")

def maslm_java():
    LOCATION_AGENT_PROMPT = txt_read_file("prompt/java/location.txt")
    ANSWER_CHANGE_AGENT_PROMPT = txt_read_file("prompt/java/answer.txt")
    FINAL_TOKEN = 0
    data = jsonl_read_file("input_dataset/test_case.jsonl")
    
    for index, CODE in enumerate(data):
        print(f"Processing COOOOOODE {index}...")
        # AST 预处理
        ast_result = extract_java_ast_structure(CODE)
        sample = {**CODE, **ast_result} # 拼接

        # MAS启动
        agents = {
            "location_library": create_agent(LOCATION_AGENT_PROMPT), 
            "answer_change": create_agent(ANSWER_CHANGE_AGENT_PROMPT,server_url= "https://mcp.context7.com/mcp",api_key="ctx7sk-97bd7e64-9cb4-477e-a13e-51c267f58e6e"),
        }

        orch = OrchestratorJava(agents, sample)

        location_result = orch.location_library()
        answer_change_result = orch.answer_change()

        # 结果写入并print
        append_to_jsonl("output_dataset/java/create_result.jsonl", answer_change_result)
        print("\n========== TOKEN USAGE SUMMARY ==========")
        for k, v in orch.token_stats.items():
            print(f"{k}: {v}")
        print("========================================\n")
        FINAL_TOKEN += orch.token_stats["total"]

    print("\n========== TOKEN USAGE SUMMARY ==========")
    print(f"FINAL_TOKEN: {FINAL_TOKEN}")
    print("========================================\n")

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "sk-rttlzkrvwxmfnolcmadlkeczxxnkmwolfprvyfnfwpfursjl"
    os.environ["OPENAI_API_BASE_URL"] = "https://api.siliconflow.cn/v1"

    # start_time = time.time()

    # Easy Python
    # maslm()

    # 判断结果
    # judge_bench()
    # result = compute_avg("output_dataset/easy_python/judge_result_change_prompt.jsonl")
    # print(result)

    # Hard Python
    # maslm_hard_python()

    # end_time = time.time()
    # print(f"Total time: {end_time - start_time} seconds")

    # Java
    maslm_java()


