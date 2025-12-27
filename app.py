import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile

# --- 页面配置 ---
st.set_page_config(page_title="论文辅助系统 (AI Research Assistant)", layout="wide")

# --- 侧边栏：设置与上传 ---
with st.sidebar:
    st.title("⚙️ 系统设置")
    api_key = st.text_input("请输入 API Key (OpenAI/DeepSeek)", type="password")
    base_url = st.text_input("API Base URL (选填)", value="https://api.openai.com/v1", help="如果是DeepSeek，填 https://api.deepseek.com")
    
    st.divider()
    st.header("📂 资料库")
    uploaded_files = st.file_uploader("上传参考文献 (PDF)", type=["pdf"], accept_multiple_files=True)
    
    process_btn = st.button("构建知识库")

# --- 核心逻辑函数 ---

def process_pdfs(files):
    """读取PDF并进行切片和向量化"""
    documents = []
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, file in enumerate(files):
        status_text.text(f"正在解析: {file.name}...")
        # 创建临时文件因为PyPDFLoader需要路径
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        documents.extend(loader.load())
        os.remove(tmp_path) # 清理临时文件
        progress_bar.progress((i + 1) / len(files))
    
    # 文本切片：学术论文需要保持上下文，chunk_size设大一点
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    return splits

def get_vector_store(splits, api_key, base_url):
    """建立向量索引"""
    # 这里默认使用 OpenAI Embeddings，也可以换成 HuggingFace 免费的
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, base_url=base_url)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def generate_academic_response(query, vectorstore, api_key, base_url):
    """生成符合顶刊风格的回答"""
    llm = ChatOpenAI(
        model_name="gpt-4o",  # 或者 deepseek-chat
        temperature=0.3, # 低温度保证严谨性
        openai_api_key=api_key,
        base_url=base_url
    )
    
    # --- 步骤 1: 检索相关信息 ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # 检索最相关的5个片段
    docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # --- 步骤 2: 顶刊写手 Agent ---
    writer_prompt = f"""
    你是一名世界顶尖的研究员，正在为 Nature/Science 级别的期刊撰写论文内容。
    
    【参考文献片段】：
    {context_text}
    
    【用户指令/想法】：
    {query}
    
    【任务】：
    请根据用户指令，严格基于参考文献片段，撰写相应的内容（如实验方案、引言或讨论）。
    
    【要求】：
    1. 逻辑严密，学术用语规范。
    2. 必须引用参考文献中的数据或观点来支持论述。
    3. 使用 LaTeX 格式编写数学公式。
    4. 结构清晰，分点论述。
    5. 如果参考文献中没有相关信息，请诚实说明，不要编造。
    
    请开始撰写：
    """
    
    with st.spinner("✍️ AI 研究员正在撰写初稿..."):
        initial_draft = llm.invoke(writer_prompt).content
        
    return initial_draft, context_text

def reviewer_critique(draft, query, api_key, base_url):
    """审稿人 Agent：挑刺模式"""
    llm = ChatOpenAI(
        model_name="gpt-4o", 
        temperature=0.7, 
        openai_api_key=api_key,
        base_url=base_url
    )
    
    reviewer_prompt = f"""
    你是一名以严厉著称的顶级期刊审稿人 (Reviewer #2)。
    
    【用户原始意图】：{query}
    
    【待审阅稿件】：
    {draft}
    
    【任务】：
    请对上述稿件进行批判性审阅。
    1. 指出逻辑漏洞。
    2. 指出语言是否不够学术（Too casual）。
    3. 指出实验设计是否缺乏控制变量（如果是实验方案）。
    4. 给出具体的修改建议。
    
    请输出你的审阅报告：
    """
    
    with st.spinner("🧐 审稿人正在极其挑剔地检查..."):
        critique = llm.invoke(reviewer_prompt).content
        
    return critique

# --- 主界面逻辑 ---

st.title("🎓 顶刊论文辅助系统")
st.markdown("### User Input: 你的想法")

user_idea = st.text_area("在此输入你的核心Idea、假设或具体要求（支持纯文本或从PDF粘贴）", height=150)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# 处理上传文件
if process_btn and uploaded_files and api_key:
    if not api_key:
        st.error("请先在左侧输入 API Key")
    else:
        try:
            splits = process_pdfs(uploaded_files)
            st.session_state.vectorstore = get_vector_store(splits, api_key, base_url)
            st.success(f"知识库构建完成！共处理 {len(splits)} 个文本片段。")
        except Exception as e:
            st.error(f"处理出错: {str(e)}")

# 生成按钮
if st.button("🚀 开始生成论文内容"):
    if not st.session_state.vectorstore:
        st.warning("请先上传参考文献并构建知识库！")
    elif not user_idea:
        st.warning("请输入你的想法！")
    else:
        # 1. 撰写
        draft, sources = generate_academic_response(user_idea, st.session_state.vectorstore, api_key, base_url)
        
        st.divider()
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📄 AI 生成初稿")
            st.markdown(draft)
            
        with col2:
            st.subheader("📚 溯源信息 (Context)")
            with st.expander("查看引用的原文片段"):
                st.markdown(sources)
        
        # 2. 审稿
        st.divider()
        st.subheader("🧐 Reviewer #2 的审阅意见")
        critique = reviewer_critique(draft, user_idea, api_key, base_url)
        st.info(critique)

        st.markdown("---")
        st.caption("提示：你可以根据审阅意见修改你的 Input，重新生成直到满意。")