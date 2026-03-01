#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026/3/1 20:01
# @Author  : SakuHx
# @File    : main.py
import importlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

# ------------------------------------------------ #

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
        self.tool_root_dir = ""
        self._tools = self._load_tools_from_dir(self.tool_root_dir)
        self.agent = create_agent(self.model, tools=self._tools)
    
    def _load_tools_from_dir(self, root_dir):
        """从指定目录自动加载工具并返回 LangChain Tool 对象列表。
    
        Args:
            root_dir (str): 存放所有工具子目录的根目录路径。
    
        Returns:
            List[Tool]: 一个包含所有成功加载的 LangChain Tool 对象的列表。
        """
        tools = []
        root_path = Path(root_dir)
        
        if not root_path.exists() or not root_path.is_dir():
            print(f"错误: 指定的工具目录不存在或不是一个目录: {root_dir}")
            return tools
        
        # 遍历根目录下的所有子目录
        for tool_subdir in root_path.iterdir():
            if not tool_subdir.is_dir():
                continue
            
            print(f"正在加载工具: {tool_subdir.name}")
            
            # 定义必需文件的路径
            desc_file_path = tool_subdir / "description.md"
            exec_file_path = tool_subdir / "executor.py"
            
            # 1. 验证必需文件是否存在
            if not (desc_file_path.exists() and exec_file_path.exists()):
                print(f"  跳过 {tool_subdir.name}: 缺少必需的 'description.md' 或 'executor.py'")
                continue
            
            # 2. 读取工具描述
            try:
                with open(desc_file_path, 'r', encoding='utf-8') as f:
                    description = f.read().strip()
            except Exception as e:
                print(f"  跳过 {tool_subdir.name}: 读取 description.md 失败 - {e}")
                continue
            
            # 3. 动态加载执行器模块
            try:
                spec = importlib.util.spec_from_file_location(
                    f"{tool_subdir.name}_executor", exec_file_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"  跳过 {tool_subdir.name}: 加载 executor.py 失败 - {e}")
                continue
            
            # 4. 寻找执行函数
            # 约定：每个 executor.py 必须有一个名为 run_tool 的函数
            # 你可以根据需要修改此约定，例如寻找特定的类或带有装饰器的函数
            if not hasattr(module, 'run_tool'):
                print(f"  跳过 {tool_subdir.name}: executor.py 中没有找到 'run_tool' 函数")
                continue
            
            run_func = getattr(module, 'run_tool')
            
            # 5. 创建 LangChain Tool 对象
            # 这里假设 run_tool 是一个可以直接使用的函数
            # 如果是类，你需要先实例化它
            # tool_name 可以从子目录名或 executor.py 中定义的变量获取
            tool_name = tool_subdir.name
            
            # --- 重要提示 ---
            # 为了让 LangChain Agent 更好地理解如何调用工具，
            # 描述中最好包含参数格式。例如：
            # description = "这是一个天气查询工具。输入: {'city': '城市名'}"
            # 或者，在 executor.py 中定义一个函数来返回 schema
            # if hasattr(module, 'get_schema'):
            #     args_schema = getattr(module, 'get_schema')()
            # else:
            #     args_schema = None
            
            # 为了简化，这里不强制要求 args_schema，但强烈建议在实际应用中提供。
            # LangChain 会尝试根据函数签名推断，但这不够可靠。
            
            try:
                # 如果你的 executor.py 中定义了参数模式 (args_schema)，请传入它
                # 否则，LangChain 会尝试推断，但手动定义更佳
                langchain_tool = Tool(
                    name=tool_name,
                    func=run_func,
                    description=description,
                    # args_schema=your_args_schema_here # 推荐在此处添加
                )
                tools.append(langchain_tool)
                print(f"  成功加载工具: {tool_name}")
            except Exception as e:
                print(f"  跳过 {tool_subdir.name}: 创建 Tool 对象失败 - {e}")
                continue
        
        return tools
    
    @property
    def tools(self):
        return self._tools


if __name__ == '__main__':
    pass
    ai = ChatAI("qwen-flash", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", key=os.getenv("API_KEY"))
    for e in ai.chat("hh，请介绍你自己，不少于100字", True):
        print(e.text, end="", flush=True)
