import os
import sys
import time
from ast_pre.ast_pre import extract_ast_structure
from utils import *
from agent.mas import Orchestrator, create_agent
from agent.judge import OrchestratorJudger
from agent.mcp_tools import ANSWER_MCP_TOOLS

def maslm():
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
            "answer_change": create_agent(ANSWER_CHANGE_AGENT_PROMPT),
            "fix_function": create_agent(FIX_FUNCTION_AGENT_PROMPT)
        }
        response = agents["location_library"].step("Use search_bing with query='周杰伦', then summarize the results.")
        print(response)
        print()
        print(response.info)
        sys.exit(0)
        
        # msgs=[
        #     BaseMessage(
        #         role_name='Assistant', 
        #         role_type=<RoleType.ASSISTANT: 'assistant'>, 
        #         meta_dict={}, 
        #         content='Alan Turing was an influential English mathematician and computer scientist, born on June 23, 1912. He is often regarded as the father of theoretical computer science due to his foundational work on algorithms and computation, notably conceptualizing the Turing machine—a model for general-purpose computers. Turing also played a pivotal role in cryptanalysis during World War II. He passed away on June 7, 1954.', 
        #         video_bytes=None, image_list=None, image_detail='auto', video_detail='low', parsed=None)] 
        # terminated=False 
        # info={'id': '019ba1e28d72d2c193af7d7052c09fa7', 'usage': {'completion_tokens': 82, 'prompt_tokens': 4692, 'total_tokens': 4774, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 0, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': None}, 'termination_reasons': ['stop'], 'num_tokens': 818, 
        #     'tool_calls': [
        #         ToolCallingRecord(
        #               tool_name='search_google', 
        #               args={'query': 'Alan Turing', 'num_result_pages': 1}, 
        #               result={'error': "Error executing tool 'search_google': Execution of function search_google failed with arguments () and {'query': 'Alan Turing', 'num_result_pages': 1}. Error: Missing or empty required API keys in environment variables: GOOGLE_API_KEY, SEARCH_ENGINE_ID.\nYou can obtain the API key from the official website"}, 
        #               tool_call_id='019ba1e269b9019922885a13538bfac8'
        #         ), 
        #         ToolCallingRecord(
        #             tool_name='search_wiki', 
        #             args={'entity': 'Alan Turing'}, 
        #             result="Alan Mathison Turing (; 23 June 1912 – 7 June 1954) was an English mathematician, computer scientist, logician, cryptanalyst, philosopher and theoretical biologist. He was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine, which can be considered a model of a general-purpose computer. Turing is widely considered to be the father of theoretical computer science.\nBorn in London, Turing was raised in southern England. He graduated from King's College, Cambridge, and in 1938, earned a doctorate degree from Princeton University.", 
        #             tool_call_id='019ba1e286053e704983e99c561918a1'
        #         )
        #     ], 
        #     'external_tool_call_requests': None
        # }
        
        # msgs=[
        #     BaseMessage(role_name='Assistant', role_type=<RoleType.ASSISTANT: 'assistant'>, meta_dict={}, 
        #                 content='I couldn\'t find any relevant results for "Alan Turing" on Baidu at this time. Let me know if you\'d like me to try another search engine or provide information from another source.', video_bytes=None, image_list=None, image_detail='auto', video_detail='low', parsed=None)] 
        # terminated=False info={'id': '019ba1e7f1984709fa3d1429eec948e3', 'usage': {'completion_tokens': 39, 'prompt_tokens': 4465, 'total_tokens': 4504, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 0, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': None}, 'termination_reasons': ['stop'], 'num_tokens': 599, 
        #                        'tool_calls': [ToolCallingRecord(tool_name='search_baidu', args={'query': 'Alan Turing', 'max_results': 5}, result={'results': []}, tool_call_id='019ba1e7f01e73434d962f08e2f2d9ae')], 'external_tool_call_requests': None}
        
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
            "answer_change": create_agent(ANSWER_CHANGE_AGENT_PROMPT),
            "fix_function": create_agent(FIX_FUNCTION_AGENT_PROMPT)
        }
    #     orch = Orchestrator(agents, sample)
    #     # MAS具体执行三个Agent,前两个不执行没法执行第三个
    #     location_result = orch.location_library()
    #     answer_change_result = orch.answer_change()
    #     fix_function_result = orch.fix_function()


    #     # 结果写入并print
    #     append_to_jsonl("output_dataset/easy_python/create_result.jsonl", fix_function_result)
    #     print("\n========== TOKEN USAGE SUMMARY ==========")
    #     for k, v in orch.token_stats.items():
    #         print(f"{k}: {v}")
    #     print("========================================\n")
    #     FINAL_TOKEN += orch.token_stats["total"]

    # print("\n========== TOKEN USAGE SUMMARY ==========")
    # print(f"FINAL_TOKEN: {FINAL_TOKEN}")
    # print("========================================\n")

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "sk-rttlzkrvwxmfnolcmadlkeczxxnkmwolfprvyfnfwpfursjl"
    os.environ["OPENAI_API_BASE_URL"] = "https://api.siliconflow.cn/v1"


    LOCATION_AGENT_PROMPT = txt_read_file("prompt/easy_python/location.txt")
    ANSWER_CHANGE_AGENT_PROMPT = txt_read_file("prompt/easy_python/answer.txt")
    FIX_FUNCTION_AGENT_PROMPT = txt_read_file("prompt/easy_python/fix_function.txt")
    JUDGE_AGENT_PROMPT = txt_read_file("prompt/easy_python/judger.txt")

    # 开始计时
    # start_time = time.time()
    maslm()
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
    # judge_bench()

    # 结束计时
    # end_time = time.time()
    # print(f"Total time: {end_time - start_time} seconds")



    # result = compute_avg("output_dataset/easy_python/judge_result.jsonl")
    # print(result)






    """
    =========================================
    hard版本打草稿
    但是并未运行
    =========================================
    """
    # LOCATION_AGENT_PROMPT = txt_read_file("prompt/hard_python/location.txt")
    # ANSWER_CHANGE_AGENT_PROMPT = txt_read_file("prompt/hard_python/answer.txt")
    # FIX_FUNCTION_AGENT_PROMPT = txt_read_file("prompt/hard_python/fix_function.txt")


    # maslm_hard_python()

