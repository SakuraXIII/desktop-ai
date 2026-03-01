#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026/3/1 20:01
# @Author  : SakuHx
# @File    : main.py
import os

# ------------------------------------------------ #

from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


class ChatAI:
    def __init__(self, model_name, provider='openai', base_url=None, key=None, temperature=0.7, timeout=30, max_tokens=1024, max_retries=3):
        super().__init__()
        self.model = init_chat_model(model=model_name, model_provider=provider, base_url=base_url, api_key=key, temperature=temperature,
                                     timeout=timeout, max_tokens=max_tokens, max_retries=max_retries)
        self.system_prompt = ""
    
    def chat(self, question, stream=False):
        conversation = [
            SystemMessage(self.system_prompt),
            HumanMessage(question)
        ]
        if stream:
            return self.model.stream(conversation)
        else:
            res = self.model.invoke(conversation)
            return res.content


class AgentAI(ChatAI):
    
    def __init__(self, model_name, provider='openai', base_url=None, key=None, temperature=0.7, timeout=30, max_tokens=1024, max_retries=3):
        super().__init__(model_name, provider, base_url, key, temperature, timeout, max_tokens, max_retries)
        tools = self._get_tools_for_dir("./skills")
        self.agent = create_agent(self.model, tools=tools)
    
    def _get_tools_for_dir(self, dir_path):
        return []


if __name__ == '__main__':
    pass
    ai = ChatAI("qwen-flash", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", key=os.getenv("API_KEY"))
    for e in ai.chat("hh，请介绍你自己，不少于100字", True):
        print(e.text, end="", flush=True)
