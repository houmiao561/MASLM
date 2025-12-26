import os
import sys
import time
from ast_pre.ast_pre import extract_ast_structure
from utils import *
from agent.mas import Orchestrator, create_agent
from agent.judge import OrchestratorJudger

def maslm():
    FINAL_TOKEN = 0
    data = jsonl_read_file("input_dataset/easy_code.jsonl")
    for index, CODE in enumerate(data):
        print(f"Processing COOOOOODE {index}...")
        # AST 预处理
        ast_result = extract_ast_structure(CODE)
        sample = {**CODE, **ast_result} # 拼接

        # MAS启动
        agents = {
            "location_library": create_agent(LOCATION_AGENT_PROMPT), 
            "answer_change": create_agent(ANSWER_CHANGE_AGENT_PROMPT),
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
    FINAL_TOKEN = 0
    data = jsonl_read_file("output_dataset/easy_python/create_result.jsonl")
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
        append_to_jsonl("output_dataset/easy_python/judge_result.jsonl", judge_result)
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


    # 读取json文件
    LOCATION_AGENT_PROMPT = txt_read_file("prompt/easy_python/location.txt")
    ANSWER_CHANGE_AGENT_PROMPT = txt_read_file("prompt/easy_python/answer.txt")
    FIX_FUNCTION_AGENT_PROMPT = txt_read_file("prompt/easy_python/fix_function.txt")
    JUDGE_AGENT_PROMPT = txt_read_file("prompt/easy_python/judger.txt")

    # 开始计时
    start_time = time.time()
    # maslm()
    # print("JUDGE")
    # print("JUDGE")
    # print("JUDGE")
    # print("JUDGE")
    # print("JUDGE")
    # print("JUDGE")
    # print("JUDGE")
    # print("JUDGE")
    # print("JUDGE")
    # print("JUDGE")
    # print("JUDGE")
    # print("JUDGE")
    judge_bench()

    # 结束计时
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")
