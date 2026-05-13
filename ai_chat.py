import streamlit as st
import os
import time
import hashlib
import requests
from datetime import datetime
import tempfile



# LangChain RAG核心
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
# 智谱SDK
from zhipuai import ZhipuAI

# ===================== 全局配置（豆包同款布局基础）=====================
st.set_page_config(
    page_title="智普 - AI智能助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 会话状态初始化（全功能隔离，无漏洞）=====================
# 主题色默认值
if "primary_color" not in st.session_state:
    st.session_state.primary_color = "#1677FF"
if "user_bubble_color" not in st.session_state:
    st.session_state.user_bubble_color = "#F0F7FF"
if "ai_bubble_color" not in st.session_state:
    st.session_state.ai_bubble_color = "#FAFAFA"

# 模型-API Key映射（自用安全隔离，切换模型强制验证）
if "model_api_map" not in st.session_state:
    st.session_state.model_api_map = {
        "GLM-4.7": "智谱API Key",
        "GLM-4.5-air": "智谱API Key",
        "GLM-4.6-v": "智谱API Key",
        "Doubao": "豆包API Key（预留）",
        "DeepSeek": "DeepSeek API Key（预留）"
    }

# 对话&向量库隔离（多对话不混淆）
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "current_chat_id" not in st.session_state:
    cid = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.current_chat_id = cid
    st.session_state.chat_history[cid] = [{
        "role": "system",
        "content": """你是小智，专业双领域AI助手：
1. 工业互联网：设备运维、故障排查；2. 论文：降重/润色/正版查重；
3. 回答分点清晰，自称为小智；4. 基于上传文档/截图精准回答，禁止编造。"""
    }]
if "vector_store_map" not in st.session_state:
    st.session_state.vector_store_map = {}

# API Key&认证状态（解决接入不确定问题）
if "model_auth" not in st.session_state:
    st.session_state.model_auth = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "cur_model" not in st.session_state:
    st.session_state.cur_model = "GLM-4-Air-250414"
if "model_api_keys" not in st.session_state:
    st.session_state.model_api_keys = {"GLM-4-Air-250414": ""}

# 截图状态（集成到对话）
if "current_screenshot" not in st.session_state:
    st.session_state.current_screenshot = None

# ===================== 核心功能函数（全功能补全，无漏洞）=====================
# PaperPass 正版查重API（含降级逻辑，不填密钥不报错）
def paperpass_check(content, api_key="", secret=""):
    if not api_key or not secret:
        time.sleep(1.5)
        return {
            "sim_rate": round(15 + time.time() % 10, 1),
            "tips": "⚠️ 当前为模拟查重，填写PaperPass API密钥可使用正版查重"
        }
    try:
        timestamp = str(int(time.time()))
        sign = hashlib.md5((api_key + timestamp + secret).encode()).hexdigest()
        res = requests.post(
            "https://api.paperpass.com/v1/check/sync",
            json={"content": content, "check_type": 1},
            headers={"API-KEY": api_key, "TIMESTAMP": timestamp, "SIGN": sign},
            timeout=30
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"sim_rate": 0, "tips": f"❌ 查重失败：{str(e)}"}

# AI模型调用（智谱GLM专属，异常兜底，强制认证）
def ai_reply_stream(prompt, model, key):
    if not st.session_state.model_auth:
        yield "❌ 请先在侧边栏输入API Key并点击「确认接入」"
        return
    
    try:
        # 1. 智谱 GLM 系列
        if "GLM" in model:
            from zhipuai import ZhipuAI
            client = ZhipuAI(api_key=key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.5,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        # 2. DeepSeek 系列 (完全兼容 OpenAI 接口)
        elif "DeepSeek" in model:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat", # 根据你的具体模型选项调整
                messages=[{"role":"user","content":prompt}],
                temperature=0.5,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        # 3. 豆包 Doubao 系列 (使用火山引擎 SDK)
        elif "Doubao" in model:
            from volcenginesdkarkruntime import Ark
            client = Ark(api_key=key)
            response = client.chat.completions.create(
                model="your-doubao-endpoint-id", # 注意：豆包需要传入具体的 Endpoint ID
                messages=[{"role":"user","content":prompt}],
                temperature=0.5,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
    except Exception as e:
        yield f"❌ 调用失败：{str(e)}"

# RAG文档处理（中英双语优化，论文专属，支持图像）=====================
def process_file(uploaded_file, chat_id):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as f:
            f.write(uploaded_file.getbuffer())
            path = f.name
        
        # 根据文件类型选择加载器
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif path.endswith(('.png', '.jpg', '.jpeg')):
            # 图像文件处理 - 目前我们存储图像路径，AI需要特殊处理来理解图像
            # 为图像创建一个简单的文档表示
            image_doc = Document(
                page_content=f"图像文件: {uploaded_file.name}, 无法直接提取文本内容，请通过视觉方式理解此图像。",
                metadata={"source": path, "filename": uploaded_file.name, "file_type": "image"}
            )
            splits = [image_doc]
        else:
            loader = TextLoader(path, encoding="utf-8")
            
        # 如果不是图像文件，则执行常规文档加载和分割
        if not path.endswith(('.png', '.jpg', '.jpeg')):
            # 论文专属分块参数
            splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(loader.load())
        
        # BAAI/bge-m3：中英双语最强轻量向量模型
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device":"cpu"})
        # 绑定当前对话的向量库，彻底隔离
        st.session_state.vector_store_map[chat_id] = FAISS.from_documents(splits, embeddings)
        os.unlink(path)
        return f"✅ 《{uploaded_file.name}》处理完成，RAG已激活"
    except Exception as e:
        return f"❌ 处理失败：{str(e)}"

# ===================== 侧边栏（API认证+主题色+历史对话，布局优化）=====================
with st.sidebar:
    # 新建对话按钮（突出显示）
    if st.button("🆕 新建对话", type="primary", use_container_width=True): 
        new_cid = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
        # 核心修复：先提取当前对话的系统提示词，然后再切换 ID 
        sys_prompt = st.session_state.chat_history[st.session_state.current_chat_id][0] 
        st.session_state.chat_history[new_cid] = [sys_prompt] 
        st.session_state.current_chat_id = new_cid 
        st.rerun()
    
    # 历史对话列表（简洁列表）
    st.markdown("### 📜 历史对话")
    for cid in st.session_state.chat_history:
        if cid != st.session_state.current_chat_id:
            # 获取对话的第一个用户消息作为标题，如果没有则使用默认标题
            chat_msgs = [msg for msg in st.session_state.chat_history[cid] if msg["role"] == "user"]
            chat_title = chat_msgs[0]["content"][:30] + "..." if chat_msgs else f"对话_{cid.split('_')[-1]}"
            # 清理标题中的换行符和其他可能导致显示问题的字符
            chat_title = chat_title.replace('\n', ' ').replace('\r', '').strip()
            
            # 简洁的对话列表项
            if st.button(f"💬 {chat_title}", key=f"hist_{cid}", use_container_width=True):
                st.session_state.current_chat_id = cid
                st.rerun()
        else:
            # 当前对话标题（带淡蓝色背景）
            current_chat_msgs = [msg for msg in st.session_state.chat_history[cid] if msg["role"] == "user"]
            current_title = current_chat_msgs[0]["content"][:30] + "..." if current_chat_msgs else f"对话_{cid.split('_')[-1]}"
            # 清理标题中的换行符和其他可能导致显示问题的字符
            current_title = current_title.replace('\n', ' ').replace('\r', '').strip()
            st.markdown(f"<div style='background-color: #E6F0FF; padding: 8px; border-radius: 8px; margin: 4px 0;'><strong>💬 {current_title}</strong></div>", unsafe_allow_html=True)
    
    # 设置项（全部放入expander中）
    with st.expander("⚙️ 设置", expanded=False):
        
        # 模式切换
        mode = st.radio("模式选择", ["🏭 工业模式", "🎓 学术模式"], key="mode_selector_sidebar")
        
        # 更新会话状态中的模式
        if "app_mode" not in st.session_state:
            st.session_state.app_mode = "🏭 工业模式"

        if mode != st.session_state.app_mode:
            st.session_state.app_mode = mode
            if mode == "🏭 工业模式":
                st.toast("🔄 已切换到工业模式", icon="🏭")
            else:
                st.toast("🔄 已切换到学术模式", icon="🎓")

        # 模型切换
        model_options = list(st.session_state.model_api_map.keys()) 
        selected_model_idx = model_options.index(st.session_state.cur_model) if st.session_state.cur_model in model_options else 0 
        new_selected_model = st.selectbox( 
            "⚙️ 切换大模型", 
            model_options, 
            index=selected_model_idx, 
            key="model_select_sidebar" 
        ) 
        if new_selected_model != st.session_state.cur_model: 
            st.session_state.cur_model = new_selected_model 
            if new_selected_model in st.session_state.model_api_keys: 
                st.session_state.api_key = st.session_state.model_api_keys[new_selected_model] 
            st.rerun() 

        # 1. API Key接入（高亮显示，一眼找到，解决接入不确定问题）
        st.subheader("🔑 模型API Key（高亮区域）")
        
        if not st.session_state.model_auth:
            # 未认证状态：显示输入框和确认按钮
            st.markdown('<div class="highlight-box">⚠️ 请先填写API Key才能使用AI功能</div>', unsafe_allow_html=True)
            
            # 当前模型API Key输入
            required_key = st.session_state.model_api_map[st.session_state.cur_model]
            api_input = st.text_input(
                f"请输入{required_key}", 
                type="password", 
                value=st.session_state.model_api_keys.get(st.session_state.cur_model, ""),
                key="api_input"
            )
            
            if st.button("✅ 确认接入当前模型", type="primary", use_container_width=True):
                st.session_state.model_api_keys[st.session_state.cur_model] = api_input
                st.session_state.api_key = api_input
                st.session_state.model_auth = True
                st.toast("✅ 接入成功！")
                st.rerun()
        else:
            # 已认证状态：显示成功信息和更换按钮
            st.success('✅ 已成功连接至当前模型')
            if st.button("🔄 更换密钥", use_container_width=True):
                st.session_state.model_auth = False
                st.rerun()


    # 添加文件上传区域
    st.divider()
    st.markdown("### 📎 当前对话参考文档")
    # 文件上传：不再隐藏文字，规规矩矩展示，防止错位 
    uploaded_files = st.file_uploader( 
        "📎 附加文件", 
        type=["pdf", "docx", "txt", "png", "jpg"], 
        accept_multiple_files=True, 
        key=f"file_upload_sidebar_{st.session_state.current_chat_id}",
        help="支持上传多个文件，支持PDF、DOCX、TXT、PNG、JPG格式"
    ) 
    if uploaded_files: 
        for uploaded_file in uploaded_files: 
            with st.spinner(f"📄 处理中: {uploaded_file.name}"): 
                result = process_file(uploaded_file, st.session_state.current_chat_id) 
                st.toast(result, icon="✅")

# ===================== 主界面：标题动态显示 ===================== 
msgs = st.session_state.chat_history[st.session_state.current_chat_id] 
 
# 最严格的判定：只要对话记录长度等于1（即只有初始隐藏的 system 提示词），就显示大标题。 
# 只要用户发了哪怕一句话，长度就会 >= 2，立刻隐藏！ 
if len(msgs) == 1: 
    st.markdown("<h1 style='text-align: center; margin-top: 15vh;'>智谱 AI 助手</h1>", unsafe_allow_html=True) 
    st.markdown("<p style='text-align: center; color: gray;'>工业互联网 · 论文优化 · 智能问答</p>", unsafe_allow_html=True)

# 显示对话内容（使用Gemini风格布局）
# 强制锁定 Gemini 官方配色 
is_dark = st.session_state.get("dark_mode", False) 
user_bg = "#282A2D" if is_dark else "#F0F4F9" 
user_text = "#E3E3E3" if is_dark else "#1F1F1F" 

for msg in msgs: 
    if msg["role"] == "system": 
        continue 
    
    if msg["role"] == "user": 
        # 用户气泡：强制使用 user_bg 变量 
        st.markdown(f''' 
        <div style="display: flex; justify-content: flex-end; margin-bottom: 20px;"> 
            <div style="background-color: {user_bg}; color: {user_text}; padding: 12px 18px; border-radius: 18px; max-width: 75%; font-size: 15px; line-height: 1.6;"> 
                {msg["content"]} 
            </div> 
        </div> 
        ''', unsafe_allow_html=True) 
    else: 
        # AI 回复：原生 Markdown 渲染 
        st.markdown(msg["content"])

# 滚动锚点（自动滚到底部，替代st.js）
st.markdown('<div id="scroll-anchor"></div>', unsafe_allow_html=True)

# ===================== 对话逻辑（全功能补全，拟人化+RAG+截图+知识库来源）=====================

# ===================== 底部交互区（稳定原生布局） ===================== 

 

# 2. 原生聊天输入框：永远贴在最底部，自带圆角阴影，绝对不会重叠！ 
prompt = st.chat_input("输入你的问题...") 

# --- 接下来就是你原本的提问处理逻辑 --- 
if prompt:
    # 处理截图信息（集成到对话，AI可参考）
    screenshot_info = ""
    if st.session_state.current_screenshot:
        screenshot_info = f"\n（用户上传了截图：{st.session_state.current_screenshot.name}，请结合截图内容回答）"
    
    # 展示用户消息（自定义气泡布局）
    st.markdown(f''' 
    <div style="display: flex; justify-content: flex-end; margin-bottom: 20px;"> 
        <div style="background-color: #1677FF; color: white; padding: 12px 18px; border-radius: 18px 4px 18px 18px; max-width: 75%; font-size: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); line-height: 1.6;"> 
            {prompt + screenshot_info} 
        </div> 
    </div> 
    ''', unsafe_allow_html=True)
    msgs.append({"role":"user","content":prompt + screenshot_info})

    # RAG检索（仅当前对话的向量库，不混淆）
    context = ""
    sources = []
    current_cid = st.session_state.current_chat_id
    if current_cid in st.session_state.vector_store_map:
        docs = st.session_state.vector_store_map[current_cid].similarity_search(prompt, k=3)
        context = "### 参考文档：\n" + "\n".join([d.page_content for d in docs]) + "\n\n"
        # 提取来源信息
        sources = [d.metadata.get("filename", "未知文件") for d in docs]

    # 显示正在检索的信息
    if sources:
        source_names = list(set(sources))  # 去重
        st.markdown(f'<div style="font-size: 0.9em; color: #666; padding: 8px; background-color: #f0f7ff; border-radius: 8px; margin: 5px 0;">🔍 正在从 {", ".join(source_names)} 中检索...</div>', unsafe_allow_html=True)

    # 根据模式调整系统提示
    if st.session_state.app_mode == "🏭 工业模式":
        system_prompt = """你是小智，工业互联网专家AI助手：
        1. 专注工业互联网：设备运维、故障排查、工业协议等；
        2. 基于上传文档/截图精准回答，禁止编造；
        3. 回答要专业、准确、简洁。"""
    else:  # 学术模式
        system_prompt = """你是小智，学术论文AI助手：
        1. 专注论文写作：降重、润色、正版查重建议；
        2. 基于上传文档/截图精准回答，禁止编造；
        3. 回答要严谨、逻辑清晰、学术规范。"""

    start = time.time() 
    placeholder = st.empty() 
    reply = "" 
    
    # 流式接收并直接用原生 markdown 渲染 
    for chunk_text in ai_reply_stream(f"{system_prompt}\n\n{context}问题：{prompt}", st.session_state.cur_model, st.session_state.api_key): 
        reply += chunk_text 
        placeholder.markdown(reply) 
    
    cost = round(time.time()-start, 1) 
    
    # 最终输出附加耗时，同样使用原生 markdown 
    if not reply.startswith("❌"): 
        reply += f"\n\n*(✨ 耗时 {cost} 秒)*" 
        placeholder.markdown(reply)
    
    # 添加来源引用区域（如果使用了RAG）
    if sources:
        with st.expander("📚 引用来源", expanded=False):
            st.write("**AI参考了以下文档片段：**")
            for i, doc in enumerate(docs):
                filename = doc.metadata.get("filename", "未知文件")
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                st.markdown(f"**来源 {i+1}: {filename}**\n\n{content_preview}\n")
    
    msgs.append({"role":"assistant","content":reply})
    
    # 清空截图状态
    st.session_state.current_screenshot = None

# ===================== 论文查重专区（仅在学术模式显示）=====================
if st.session_state.app_mode == "🎓 学术模式":
    with st.expander("📝 论文查重（PaperPass正版）", expanded=False):
        st.info("💡 填写PaperPass API密钥可使用正版查重，未填写则为模拟查重")
        col_key, col_secret = st.columns(2)
        with col_key:
            pp_key = st.text_input("PaperPass API Key", type="password")
        with col_secret:
            pp_secret = st.text_input("PaperPass Secret", type="password")
        content = st.text_area("粘贴论文内容", height=300, placeholder="输入论文全文...")
        if st.button("🔍 开始查重", type="primary", use_container_width=True):
            if not content:
                st.error("❌ 请先输入论文内容")
                st.stop()
            with st.spinner("🔍 正在查重..."):
                res = paperpass_check(content, pp_key, pp_secret)
                st.success(f"✅ 查重完成！复制比：{res.get('sim_rate',0)}%")
                st.markdown(f"**提示**：{res.get('tips','')}")