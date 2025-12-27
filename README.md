# 🎓 顶刊论文辅助系统 (Paper Assistant)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green)](https://www.langchain.com/)

这是一个基于 **RAG (检索增强生成)** 技术的学术论文辅助工具。它不仅能帮助你整理本地 PDF 文献，还能模拟**顶刊作者**帮你撰写初稿，并由**模拟审稿人 (Reviewer #2)** 对内容进行“毒舌”评审，助你打磨出高质量的学术内容。

## ✨ 核心功能

* **📚 本地知识库构建**：支持上传多个 PDF 文献，自动解析、切片并建立向量索引（FAISS），数据完全本地化处理，保护隐私。
* **✍️ 顶刊写手 Agent**：根据你的核心 Idea，结合上传的参考文献，模拟 Nature/Science 风格撰写引言、实验方案或讨论，并自动生成引用。
* **🧐 审稿人 Agent**：内置严厉的“Reviewer #2”角色，对生成的初稿进行批判性审阅，指出逻辑漏洞、语言问题和实验设计缺陷。
* **⚙️ 灵活的 API 支持**：支持 OpenAI 官方 Key 以及各类中转/免费 API（如 ChatAnywhere, DeepSeek 等）。

## 🛠️ 快速开始 (小白教程)

### 第一步：环境准备
确保你的电脑上安装了 Python (建议版本 3.8 或以上)。如果没有，请去 [Python 官网](https://www.python.org/downloads/) 下载并安装。

### 第二步：下载代码
将本项目的所有文件下载到你电脑的一个文件夹中。

### 第三步：安装依赖库
1. 打开电脑的终端 (Windows 用户搜索 `cmd` 或 `PowerShell`，Mac 用户打开 `Terminal`)。
2. 使用 `cd` 命令进入到项目文件夹路径。例如：
   ```bash
   cd D:\我的项目\Paper-Assistant
