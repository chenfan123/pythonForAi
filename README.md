# pythonForAi

Python for AI 学习项目，采用企业常见的 `src` 布局与可安装包结构。

---

## 目录总览

```
pythonForAi/
├── .github/                 # GitHub 相关配置（CI 流水线等）
├── .vscode/                 # VS Code / Cursor 编辑器配置
├── lessons/                 # 课程示例脚本（学习用，薄层调用）
├── src/                     # 源代码根目录（src 布局）
│   └── python_for_ai/       # 主 Python 包
├── tests/                   # 单元测试
├── .env                     # 本地环境变量（含 API Key，不提交 Git）
├── .env.example             # 环境变量模板（可提交，供他人参考）
├── .gitignore               # Git 忽略规则
├── Dockerfile               # Docker 镜像构建文件
├── Makefile                 # 常用开发命令快捷入口
├── pyproject.toml           # 项目元数据、依赖、工具配置（现代 Python 标准）
├── pyrightconfig.json       # Pyright 类型检查配置
├── README.md                # 项目说明（本文件）
├── run.sh                   # 使用 .venv 运行 Python 的简易脚本
└── .venv/                   # 虚拟环境（make install 后生成，不提交 Git）
```

---

## 文件夹说明

### `src/` — 源代码目录（src 布局）

企业项目通常把可安装的包放在 `src/` 下，而不是直接放在项目根目录。好处是：

- 开发和测试时不会「误 import」到未安装的源码副本
- 包结构与课程脚本、测试代码清晰分离
- 与 `pip install -e .` 可编辑安装配合良好

本项目中，`src/` 下只有主包 `python_for_ai/`。

---

### `src/python_for_ai/` — 主 Python 包

这是项目的**核心业务代码**，安装后可在任意位置 `import python_for_ai`。

| 文件 | 作用 |
|------|------|
| `__init__.py` | 包入口，对外导出 `get_llm_response`、`print_llm_response`，方便 `from python_for_ai import ...` |
| `config.py` | 配置管理：定位项目根目录、加载 `.env`、读取 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL`、定义模型名等常量 |
| `cli.py` | 命令行入口；安装包后可通过 `python-for-ai "你的问题"` 调用 LLM |
| `llm/` | LLM 相关子模块目录 |
| `llm/__init__.py` | 子包入口，导出 LLM 服务函数 |
| `llm/client.py` | 创建并缓存 OpenAI 兼容客户端（连接 API 的「连接层」） |
| `llm/service.py` | LLM 业务逻辑：组装 prompt、调用模型、返回或打印结果（「业务层」） |

**分层关系：**

```
lessons / cli  →  llm/service.py  →  llm/client.py  →  config.py  →  .env
   （调用方）        （业务）            （客户端）         （配置）
```

---

### `lessons/` — 课程示例脚本

存放每一课的学习脚本，**只写课程相关逻辑**，通用能力放在 `python_for_ai` 包里。

| 文件 | 作用 |
|------|------|
| `01_prompt.py` | 第一课示例：调用 `print_llm_response` 发送简单 prompt |

后续新增课程时，在此目录添加 `02_xxx.py`、`03_xxx.py` 等即可。

---

### `tests/` — 单元测试

使用 **pytest** 运行，通过 mock 避免测试时真实调用 API。

| 文件 | 作用 |
|------|------|
| `conftest.py` | pytest 全局 fixture；每个测试前后重置 LLM 客户端缓存，避免测试间相互影响 |
| `test_config.py` | 测试项目根目录、`.env` 路径等配置是否正确 |
| `test_llm_service.py` | 测试 `get_llm_response` 的入参校验与返回逻辑（mock API） |

---

### `.github/` — GitHub 自动化

| 路径 | 作用 |
|------|------|
| `workflows/ci.yml` | CI 流水线：在 push/PR 时自动跑 ruff、pyright、pytest（Python 3.11 / 3.12） |

---

### `.vscode/` — 编辑器配置

| 文件 | 作用 |
|------|------|
| `settings.json` | 指定本项目使用 `.venv` 里的 Python，并把 `src/` 加入分析路径，减少 IDE 报「找不到模块」 |

仅在 VS Code / Cursor 中生效，不影响代码运行。

---

### `.venv/` — 虚拟环境（本地生成）

- 由 `make install` 或 `python -m venv .venv` 创建
- 存放本项目专用的 Python 和第三方包（`openai`、`pytest` 等）
- **不要提交到 Git**（已在 `.gitignore` 中忽略）
- 激活后：`source .venv/bin/activate`，终端里的 `python` 即指向此环境

---

## 根目录文件说明

### 配置与依赖

| 文件 | 作用 |
|------|------|
| `pyproject.toml` | **项目核心配置文件**：包名、版本、运行时依赖（`openai`、`python-dotenv`）、开发依赖（`pytest`、`ruff`、`pyright`）、CLI 入口 `python-for-ai`、以及 pytest / ruff / pyright 的工具配置 |
| `pyrightconfig.json` | Pyright 静态类型检查配置：扫描 `src`、`tests`、`lessons`，关联 `.venv` |
| `.env.example` | 环境变量**模板**；复制为 `.env` 后填入真实 API Key |
| `.env` | **本地私密配置**（API Key 等）；由你自己创建，**切勿提交** |
| `.gitignore` | 规定哪些文件不进入 Git（`.venv`、`.env`、缓存、`__pycache__` 等） |

### 构建与运行

| 文件 | 作用 |
|------|------|
| `Makefile` | 常用命令集合：`install`、`test`、`lint`、`format`、`run-lesson`、`clean` |
| `run.sh` | 快捷脚本：自动使用项目根目录下的 `.venv/bin/python` 执行命令，等价于 `./run.sh lessons/01_prompt.py` |
| `Dockerfile` | 构建 Docker 镜像：安装 `python_for_ai` 包，默认入口为 `python-for-ai` 命令 |

### 文档

| 文件 | 作用 |
|------|------|
| `README.md` | 项目说明文档（本文件） |

---

## 快速开始

```bash
# 1. 安装（创建 .venv + 安装包与开发工具）
make install

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY

# 4. 运行第一课
python lessons/01_prompt.py

# 或使用命令行工具
python-for-ai "Hello, how are you?"
```

---

## 开发命令

```bash
make test        # 运行 pytest
make lint        # ruff 代码检查 + pyright 类型检查
make format      # ruff 自动格式化
make run-lesson  # 运行 lessons/01_prompt.py
make clean       # 清理构建产物与缓存
make help        # 查看所有命令
```

---

## 常见问题

### 为什么要用 `.venv`，不直接用系统 `python3`？

系统 Python 通常不允许随意 `pip install`（macOS 的 externally-managed-environment），且不同项目的依赖版本可能冲突。每个项目独立 `.venv` 是企业标准做法。

### 为什么 `lessons` 里能 `from python_for_ai import ...`？

执行 `make install` 时会运行 `pip install -e ".[dev]"`（可编辑安装），把 `src/python_for_ai` 注册到当前虚拟环境中，因此无需手动设置 `PYTHONPATH`。

### `.env` 和 `config.py` 分别做什么？

- `.env`：存放密钥和本地配置（不提交 Git）
- `config.py`：在代码里读取 `.env`、提供统一的配置访问接口

### 新增一课怎么写？

在 `lessons/` 下新建脚本，从包中导入需要的函数即可：

```python
from python_for_ai import print_llm_response

print_llm_response("你的 prompt")
```
