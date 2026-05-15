# 智普AI助手

一个功能丰富的AI聊天应用程序，集成了多种AI模型和实用功能。

## 功能特性

- 🤖 支持多种AI模型（智谱GLM系列、DeepSeek等）
- 📄 RAG文档处理（支持PDF、DOCX、TXT、图片格式）
- 🔍 工业模式与学术模式切换
- 💬 多对话历史管理
- 📝 论文查重功能（模拟版）
- 🎨 自定义主题色彩

## 安装步骤

1. 克隆项目
   ```bash
   git clone <your-repo-url>
   cd ai-assistant

2.创建虚拟环境并安装依赖
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
pip install -r requirements.txt

3.启动应用
streamlit run ai_chat.py
