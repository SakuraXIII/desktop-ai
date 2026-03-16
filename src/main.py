#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026/3/1 20:01
# @Author  : SakuHx
# @File    : main.py
import importlib
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from loguru import logger

# ------------------------------------------------ #

load_dotenv()


class SkillLoader:
    def __init__(self, skills_dir: str):
        self.skills = {}
        for f in sorted(Path(skills_dir).rglob("SKILL.md")):
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body}
    
    def get_descriptions(self) -> str:
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "")
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)
    
    def get_content(self, name: str) -> str:
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'."
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"
    
    def _parse_frontmatter(self, text: str) -> tuple:
        """Parse YAML frontmatter between --- delimiters."""
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()
        return meta, match.group(2).strip()


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
    
    def __init__(
            self, model_name, provider='openai', base_url=None, key=None, temperature=0.7, timeout=30,
            max_tokens=1024, max_retries=3
    ):
        super().__init__(model_name, provider, base_url, key, temperature, timeout, max_tokens, max_retries)
        self.tool_root_dir = "tools"
        self.skill_root_dir = 'skills'
        self._tools = self._load_tools_from_dir()
        self.skills = SkillLoader(self.skill_root_dir)
        self.agent = create_agent(self.model, tools=self._tools)
    
    def _load_tools_from_dir(self):
        """从指定目录自动加载工具并返回 LangChain Tool 对象列表。

        Returns:
            List: 一个包含所有成功加载的 LangChain Tool 对象的列表。
        """
        tools = []
        root_path = Path(self.tool_root_dir)
        
        if not root_path.exists() or not root_path.is_dir():
            logger.error(f"错误: 指定的工具目录不存在或不是一个目录: {self.tool_root_dir}")
            return tools
        
        # 遍历根目录下的所有子目录
        for tool_module in root_path.iterdir():
            if not tool_module.is_dir():
                continue
            
            logger.debug(f"正在加载工具: {tool_module.name}")
            
            # 动态加载执行器模块
            try:
                module = importlib.import_module(f".{tool_module.name}", package=root_path.stem)
                tools.extend(module.tools or [])
                logger.debug([tool.name for tool in module.tools])
            
            except Exception as e:
                print(f"跳过 {tool_module.name}: 加载失败 - {e}")
                continue
        
        return tools
    
    def chat(self, question, stream=False):
        conversation = HumanMessage(question)
        
        if stream:
            return self.agent.stream(conversation)
        else:
            res = self.agent.invoke({"messages": [{"role": "user", "content": f"{question}"}]})
            return res


if __name__ == '__main__':
    pass
    ai = ChatAI("qwen-flash", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", key=os.getenv("API_KEY"))
    print(ai.chat("hh，请介绍你自己，不少于100字"))
