import io
import numpy as np
import streamlit as st
from openai import OpenAI
from docx import Document

st.set_page_config(page_title="林草智答", page_icon="🌲", layout="wide")

st.title("🌲 林草智答")
st.caption("面向林草基层人员的智能问答助手 · 基于规程知识库的检索问答")

# ---------------- 侧边栏：设置与知识库 ----------------
with st.sidebar:
    st.header("① 模型设置")
    api_key = st.text_input("大模型 API Key", type="password", help="在阿里云百炼控制台获取")
    base_url = st.text_input("接口地址", value="https://dashscope.aliyuncs.com/compatible-mode/v1")
    chat_model = st.text_input("对话模型", value="qwen-plus")
    emb_model = st.text_input("向量模型", value="text-embedding-v3")

    st.divider()
    st.header("② 上传知识库文档")
    uploads = st.file_uploader(
        "上传规程 / 方案 / 手册",
        type=["docx", "txt", "md"],
        accept_multiple_files=True,
    )
    build = st.button("构建知识库", type="primary")

# ---------------- 工具函数 ----------------
def read_any(f):
    data = f.read()
    name = f.name.lower()
    if name.endswith(".docx"):
        d = Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    return data.decode("utf-8", errors="ignore")

def split_text(text, size=400, overlap=80):
    text = text.replace("\r\n", "\n")
    chunks, i = [], 0
    while i < len(text):
        seg = text[i:i + size]
        if seg.strip():
            chunks.append(seg)
        i += size - overlap
    return chunks

def get_embedding(client, model, texts):
    resp = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def retrieve(client, model, question, chunks, embs, topk=4):
    qv = get_embedding(client, model, [question])[0]
    norms = np.linalg.norm(embs, axis=1) * np.linalg.norm(qv) + 1e-8
    sims = (embs @ qv) / norms
    idx = np.argsort(-sims)[:topk]
    return [(chunks[i], float(sims[i])) for i in idx]

def ask_ai(client, model, question, contexts):
    ctx = "\n\n".join(f"[资料{i+1}]（来源：{c[0]}）\n{c[1]}" for i, (c, _) in enumerate(contexts))
    prompt = (
        "你是林草行业技术助手。请仅依据下面提供的资料回答用户问题；"
        "如果资料中没有答案，请如实说明并建议转人工。"
        "回答要简洁、专业，并在末尾用【出处】标注用到的资料编号。\n\n"
        f"资料：\n{ctx}\n\n用户问题：{question}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

HANDOFF_WORDS = ["投诉", "人工", "转客服", "举报", "找领导"]

# ---------------- 构建知识库 ----------------
if build:
    if not api_key:
        st.warning("请先填写大模型 API Key")
    elif not uploads:
        st.warning("请先上传至少一份文档")
    else:
        client = OpenAI(api_key=api_key, base_url=base_url)
        chunks = []
        with st.spinner("正在读取并切分文档..."):
            for f in uploads:
                text = read_any(f)
                for c in split_text(text):
                    chunks.append((f.name, c))
        with st.spinner(f"正在向量化 {len(chunks)} 个片段..."):
            embs = get_embedding(client, emb_model, [c for _, c in chunks])
        st.session_state["chunks"] = chunks
        st.session_state["embs"] = embs
        st.success(f"知识库已就绪：共 {len(chunks)} 个片段，来自 {len(uploads)} 份文档")

# ---------------- 问答 ----------------
if "chunks" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("输入问题，例如：古树群如何认定？")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if any(w in question for w in HANDOFF_WORDS):
                msg = "这个问题我先为你转接人工客服，请稍候。"
                st.warning(msg)
            elif not api_key:
                msg = "请先在左侧填写大模型 API Key。"
                st.error(msg)
            else:
                client = OpenAI(api_key=api_key, base_url=base_url)
                with st.spinner("检索知识库并生成答案..."):
                    contexts = retrieve(client, emb_model, question,
                                        st.session_state["chunks"], st.session_state["embs"])
                    msg = ask_ai(client, chat_model, question, contexts)
                    st.markdown(msg)
                    with st.expander("查看引用来源"):
                        for c, s in contexts:
                            st.markdown(f"**{c[0]}**　相关度 {s:.2f}")
                            st.caption(c[1][:200] + " ...")
        st.session_state["messages"].append({"role": "assistant", "content": msg})
else:
    st.info("请先在左侧填写 API Key、上传文档，然后点击「构建知识库」。")
