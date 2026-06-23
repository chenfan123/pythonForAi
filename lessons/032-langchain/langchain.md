# LangChain

LangChain: 构建简单的智能体应用，无需复杂的编排需求，能力抽象层（LLM、Tool、Message 标准化），负责有什么能力
LangGraph: 核心思想是将智能体内部抽象成一张有向图，执行与编排层（状态机、工作流、多 agent 系统），负责怎么跑
Deep Agent: 增加了规划能力、文件系统、子 agent 等高级功能。
LangSmith: 用于跟踪、记录和分析智能体在运行过程中的完整调用链路

#### 前置知识

##### 虚拟环境设置

1. 使用 venv：python 自带的 `python -m venv .venv`,venv 不负责安装新的 python 解释器，只能基于安装好的 python 创建虚拟环境。也不负责管理 CUDA、系统库等非 python 依赖。
2. 使用 uv:适合纯 python 项目
3. 使用 conda:适合 python + 非 python 依赖的项目

##### conda 创建环境

1. conda create --name 环境名称 python=3.13.12
2. conda env list # 查看环境
3. conda init # 初始化虚拟环境
4. conda activate 环境名称 # 激活环境

##### 下载 langchain

1. conda install langchain # 安装 langchain
2. conda install -c conda-forge langchain # 指定频道
3. conda update langchain # 更新 langchain
4. conda uninstall langchain # 卸载 langchain
5. conda list # 列出所有安装的包

#### 应用场景

##### RAG

流程图

![RAG 流程图](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/2bafbf3d179841049a3a6bb9d7e7ab7d~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5pyd5pquXw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzk0OTkxOTU3ODM2NTc1MSJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1782317024&x-orig-sign=w%2BQi9ZO3gmPzct5NOjD8NAcm0hA%3D)

##### Agent

![Agent 流程图](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5dd69453c8884d9cbee8be1cd34d6309~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5pyd5pquXw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzk0OTkxOTU3ODM2NTc1MSJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1782317177&x-orig-sign=pPWO30IuPgZkXB%2B7W%2F3UT9GzEvI%3D)
