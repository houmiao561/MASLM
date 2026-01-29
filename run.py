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
    data = jsonl_read_file("input_dataset/easy_code.jsonl")
    data = data[367:]
    for index, CODE in enumerate(data):
        print(f"Processing COOOOOODE {index}...")
        # AST 预处理
        ast_result = extract_ast_structure(CODE)
        sample = {**CODE, **ast_result} # 拼接

        # MAS启动
        agents = {
            "location_library": create_agent(LOCATION_AGENT_PROMPT), 
            # "answer_change": create_agent(ANSWER_CHANGE_AGENT_PROMPT,server_url= "https://mcp.context7.com/mcp",api_key="ctx7sk-97bd7e64-9cb4-477e-a13e-51c267f58e6e"), # hm的
            "answer_change": create_agent(ANSWER_CHANGE_AGENT_PROMPT,server_url= "https://mcp.context7.com/mcp",api_key="ctx7sk-2d508d17-c205-48b6-a173-db2906d9d565"), # 114的Google
            "fix_function": create_agent(FIX_FUNCTION_AGENT_PROMPT)
        }

        orch = Orchestrator(agents, sample)


        # MAS具体执行三个Agent,前两个不执行没法执行第三个
        location_result = orch.location_library()
        answer_change_result = orch.answer_change()
        fix_function_result = orch.fix_function()


        # 结果写入并print
        append_to_jsonl("output_dataset/easy_python/create_result_ALL.jsonl", fix_function_result)
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
    data = jsonl_read_file("/Users/houmiao/Desktop/MASLM/output_dataset/easy_python/create_result_ALL.jsonl")
    # print(len(data))
    # sys.exit()
    # 选择前50个
    # data = data[300:]
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
        append_to_jsonl("output_dataset/easy_python/judge_result_ALL_origin_prompt.jsonl", judge_result)
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
    data = jsonl_read_file("input_dataset/hard_code.jsonl")
    data = data[181:]
    for index, CODE in enumerate(data):
        print(f"Processing COOOOOODE {index}...")
        # AST 预处理
        ast_result = extract_ast_structure(CODE)
        sample = {**CODE, **ast_result} # 拼接

        # MAS启动
        agents = {
            "location_library": create_agent(LOCATION_AGENT_PROMPT), 
            # "answer_change": create_agent(ANSWER_CHANGE_AGENT_PROMPT,server_url= "https://mcp.context7.com/mcp",api_key="ctx7sk-97bd7e64-9cb4-477e-a13e-51c267f58e6e"),
            # "answer_change": create_agent(ANSWER_CHANGE_AGENT_PROMPT,server_url= "https://mcp.context7.com/mcp",api_key="ctx7sk-2d508d17-c205-48b6-a173-db2906d9d565"),
            "answer_change": create_agent(ANSWER_CHANGE_AGENT_PROMPT,server_url= "https://mcp.context7.com/mcp",api_key="ctx7sk-152519e7-c899-49b9-8d91-a1a125090b9c"),
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

def judge_hard_python_bench():
    JUDGE_AGENT_PROMPT = txt_read_file("prompt/hard_python/judger.txt")
    FINAL_TOKEN = 0
    data = jsonl_read_file("/Users/houmiao/Desktop/MASLM/output_dataset/hard_python/create_result.jsonl")
    data = data[178:]
    for index, CODE in enumerate(data):
        print(f"JUDGE COOOOOODE {index}...")
        # MAS启动
        agents = {
            "judger": create_agent(JUDGE_AGENT_PROMPT), 
        }
        orch = OrchestratorJudger(agents, CODE)
        # MAS具体执行三个Agent
        
        judge_result = orch.judge_hardpy_function()
        print(judge_result)

        # 结果写入并print
        append_to_jsonl("output_dataset/hard_python/judge_result.jsonl", judge_result)
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
    data = jsonl_read_file("input_dataset/java_code.jsonl")
    data = data[150:]
    # data = data[50:52]
    # 这里是两个，jsonl中的51和52
    # print(data)
    # sys.exit()
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
        append_to_jsonl("output_dataset/java/create_result_ALL_si_flow.jsonl", answer_change_result)
        print("\n========== TOKEN USAGE SUMMARY ==========")
        for k, v in orch.token_stats.items():
            print(f"{k}: {v}")
        print("========================================\n")
        FINAL_TOKEN += orch.token_stats["total"]

    print("\n========== TOKEN USAGE SUMMARY ==========")
    print(f"FINAL_TOKEN: {FINAL_TOKEN}")
    print("========================================\n")

def judge_java_bench():
    JUDGE_AGENT_PROMPT = txt_read_file("prompt/java/judger.txt")
    FINAL_TOKEN = 0
    data = jsonl_read_file("/Users/houmiao/Desktop/MASLM/output_dataset/java/create_result_ALL_si_flow.jsonl")
    data = data[150:]
    for index, CODE in enumerate(data):
        print(f"JUDGE COOOOOODE {index}...")
        # MAS启动
        agents = {
            "judger": create_agent(JUDGE_AGENT_PROMPT), 
        }
        orch = OrchestratorJudger(agents, CODE)
        # MAS具体执行三个Agent
        
        judge_result = orch.judge_java_function()

        # 结果写入并print
        append_to_jsonl("output_dataset/java/judge_result_ALL_si_flow.jsonl", judge_result)
        print("\n========== TOKEN USAGE SUMMARY ==========")
        for k, v in orch.token_stats.items():
            print(f"{k}: {v}")
        print("========================================\n")
        FINAL_TOKEN += orch.token_stats["total"]

    print("\n========== TOKEN USAGE SUMMARY ==========")
    print(f"FINAL_TOKEN: {FINAL_TOKEN}")
    print("========================================\n")


if __name__ == "__main__":
    # os.environ["OPENAI_API_KEY"] = "sk-rttlzkrvwxmfnolcmadlkeczxxnkmwolfprvyfnfwpfursjl" # 自己的硅基流动
    # os.environ["OPENAI_API_KEY"] = "sk-lhwlcbtphqsthcxfgfalmmphkjjvvttvdmhjkbhzryosigbc" # lxb的硅基流动
    os.environ["OPENAI_API_KEY"] = "sk-zvnhdpxqzhxfvctazhdtdwpwulmzzvuwadwervloujwdqylk" # 114的硅基流动
    # os.environ["OPENAI_API_KEY"] = "sk-c44d67fe2596419d8ece2b648c2064c4" # DS官网

    # os.environ["OPENAI_API_BASE_URL"] = "https://api.deepseek.com/"

    os.environ["OPENAI_API_BASE_URL"] = "https://api.siliconflow.cn/v1"

    # start_time = time.time()

    # Easy Python
    # maslm()
    # ctx7sk-2d508d17-c205-48b6-a173-db2906d9d565

    # 判断结果
    # judge_bench()
    result_easy = compute_avg("output_dataset/easy_python/judge_result_ALL.jsonl")
    print(f"""Easy Python: {result_easy}\n""")

    # Hard Python
    # maslm_hard_python()

    # 判断结果
    # judge_hard_python_bench()
    result_hard = compute_avg("output_dataset/hard_python/judge_result.jsonl")
    print(f"""Hard Python: {result_hard}\n""")
    

    # Java
    # maslm_java()

    # 判断结果
    # judge_java_bench()
    result_java = compute_avg("output_dataset/java/judge_result_ALL_si_flow.jsonl")
    print(f"""Java: {result_java}\n""")

    # end_time = time.time()
    # print(f"Total time: {end_time - start_time} seconds")
