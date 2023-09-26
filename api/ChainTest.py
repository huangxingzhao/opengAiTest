#-*- coding:utf-8 -*-
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.chains import LLMRequestsChain
from langchain.chains import TransformChain, SequentialChain

template = """在 >>> 和 <<< 直接是来自Google的原始搜索结果.
请把对于问题 '{query}' 的答案从里面提取出来，如果里面没有相关信息的话就说 "找不到"
请使用如下格式：
Extracted:<answer or "找不到">
>>> {requests_result} <<<
Extracted:"""

PROMPT = PromptTemplate(
    input_variables=["query", "requests_result"],
    template=template,
)
requests_chain = LLMRequestsChain(llm_chain=LLMChain(
    llm=OpenAI(temperature=0, openai_api_base="https://api.rcouyi.com/v1",
               openai_api_key="sk-TiEAWDcxuOdvorJkF9394dA95b124d04Af5543090564FbE2"), prompt=PROMPT))
question = "how is beijing weather like today"
inputs = {
    "query": question,
    "url": "https://www.google.com/search?q=" + question.replace(" ", "+")
}


def parse_weather_info(weather_info)->dict:
    print(weather_info)
    # 将天气信息拆分成不同部分
    parts = weather_info.split(': ')[1].split(',')

    # 解析天气
    weather = parts[0].strip()

    # 解析温度范围，并提取最小和最大温度
    temperature = parts[0].strip().replace('℃', '')

    # 解析风向和风力
    rain = parts[1]
    wet = parts[2]
    wind = parts[3]
    weather_dict = {
        'weather': weather,
        'temperature': temperature,
        'rain': rain,
        'wet' : wet,
        'win' : wind,

    }
    return weather_dict

def transform_func(inputs: dict) -> dict:
    text = inputs["output"]
    return {"weather_info" : parse_weather_info(text)}

transformation_chain = TransformChain(input_variables=["output"],
                                      output_variables=["weather_info"], transform=transform_func)

final_chain = SequentialChain(chains=[requests_chain, transformation_chain],
                              input_variables=["query", "url"])
final_result = final_chain.run(inputs)

print(final_result)