# -*- coding: utf-8 -*-
"""林草智答 · 轻量版
只依赖 Flask，其余全部使用 Python 标准库（不编译、不装 numpy/pyarrow）。
"""
import io
import json
import math
import zipfile
import urllib.request
from xml.etree import ElementTree as ET

from flask import Flask, request, jsonify, Response

app = Flask(__name__)

DEFAULT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

STATE = {
    "api_key": "",
    "base_url": DEFAULT_BASE,
    "chat_model": "qwen-plus",
    "emb_model": "text-embedding-v3",
    "chunks": [],   # [(来源文件名, 文本片段), ...]
    "embs": [],     # [向量, ...]
}

HANDOFF_WORDS = ["投诉", "人工", "转客服", "举报", "找领导"]

WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


# ---------------- 调用大模型（标准库 urllib，无需 openai 包） ----------------
def http_post_json(url, payload, api_key):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def embed_texts(texts):
    url = STATE["base_url"].rstrip("/") + "/embeddings"
    payload = {"model": STATE["emb_model"], "input": texts}
    res = http_post_json(url, payload, STATE["api_key"])
    return [d["embedding"] for d in res["data"]]


def chat_answer(question, contexts):
    url = STATE["base_url"].rstrip("/") + "/chat/completions"
    ctx = "\n\n".join(
        "[资料%d]（来源：%s）\n%s" % (i + 1, c[0], c[1]) for i, c in enumerate(contexts)
    )
    prompt = (
        "你是林草行业技术助手。请只依据下面提供的资料回答用户问题；"
        "如果资料中没有答案，请如实说明并建议转人工。"
        "回答要简洁、专业，并在末尾用【出处】标注用到的资料编号。\n\n"
        "资料：\n" + ctx + "\n\n用户问题：" + question
    )
    payload = {
        "model": STATE["chat_model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    res = http_post_json(url, payload, STATE["api_key"])
    return res["choices"][0]["message"]["content"]


# ---------------- 读取文档（标准库解析 docx） ----------------
def docx_to_text(data):
    """docx 本质是 zip + xml，用标准库直接抽取文字。"""
    z = zipfile.ZipFile(io.BytesIO(data))
    root = ET.fromstring(z.read("word/document.xml"))
    lines = []
    for p in root.iter(WORD_NS + "p"):
        s = "".join(t.text or "" for t in p.iter(WORD_NS + "t"))
        if s.strip():
            lines.append(s)
    return "\n".join(lines)


def split_text(text, size=400, overlap=80):
    text = text.replace("\r\n", "\n")
    out, i = [], 0
    while i < len(text):
        seg = text[i:i + size]
        if seg.strip():
            out.append(seg)
        i += size - overlap
    return out


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-8)


# ---------------- 路由 ----------------
@app.route("/")
def index():
    return Response(HTML, mimetype="text/html; charset=utf-8")


@app.route("/build", methods=["POST"])
def build():
    key = (request.form.get("api_key") or "").strip()
    base = (request.form.get("base_url") or "").strip() or DEFAULT_BASE
    STATE["chat_model"] = (request.form.get("chat_model") or "").strip() or "qwen-plus"
    STATE["emb_model"] = (request.form.get("emb_model") or "").strip() or "text-embedding-v3"
    files = request.files.getlist("files")

    if not key:
        return jsonify({"ok": False, "msg": "请先填写 API Key"})
    if not files:
        return jsonify({"ok": False, "msg": "请先选择至少一份文档"})

    STATE["api_key"] = key
    STATE["base_url"] = base

    chunks = []
    for f in files:
        raw = f.read()
        name = f.filename or "文档"
        if name.lower().endswith(".docx"):
            try:
                text = docx_to_text(raw)
            except Exception:
                text = ""
        else:
            text = raw.decode("utf-8", errors="ignore")
        for c in split_text(text):
            chunks.append((name, c))

    if not chunks:
        return jsonify({"ok": False, "msg": "文档里没有读到内容"})

    try:
        embs = []
        batch = 10
        for i in range(0, len(chunks), batch):
            part = [c for _, c in chunks[i:i + batch]]
            embs.extend(embed_texts(part))
    except Exception as e:
        return jsonify({"ok": False, "msg": "向量化失败，请检查 API Key 与网络：" + str(e)})

    STATE["chunks"] = chunks
    STATE["embs"] = embs
    return jsonify({"ok": True, "msg": "知识库已就绪：%d 个片段，来自 %d 份文档" % (len(chunks), len(files))})


@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True, silent=True) or {}
    q = (body.get("q") or "").strip()
    if not q:
        return jsonify({"ok": False, "msg": "请输入问题"})
    if not STATE["chunks"]:
        return jsonify({"ok": False, "msg": "请先构建知识库"})
    if any(w in q for w in HANDOFF_WORDS):
        return jsonify({"ok": True, "answer": "这个问题我先为你转接人工客服，请稍候。", "sources": []})

    try:
        qv = embed_texts([q])[0]
        scored = sorted(((cosine(qv, e), i) for i, e in enumerate(STATE["embs"])), reverse=True)
        top = scored[:4]
        contexts, sources = [], []
        for sc, i in top:
            src, txt = STATE["chunks"][i]
            contexts.append((src, txt))
            sources.append({"source": src, "snippet": txt[:200], "score": round(float(sc), 3)})
        ans = chat_answer(q, contexts)
    except Exception as e:
        return jsonify({"ok": False, "msg": "请求模型失败：" + str(e)})

    return jsonify({"ok": True, "answer": ans, "sources": sources})


HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>林草智答</title>
<style>
  *{box-sizing:border-box;}
  body{margin:0;font-family:"Microsoft YaHei",sans-serif;background:#f5f7f5;color:#222;}
  .wrap{display:flex;height:100vh;}
  aside{width:300px;background:#1f3d2b;color:#fff;padding:16px;overflow:auto;}
  aside h3{font-size:14px;margin:18px 0 8px;opacity:.85;}
  aside input{width:100%;padding:8px;margin-bottom:8px;border:none;border-radius:6px;font-size:13px;}
  aside button{width:100%;padding:10px;background:#3fa66a;color:#fff;border:none;border-radius:6px;font-size:14px;cursor:pointer;}
  aside button:hover{background:#358c59;}
  #status{margin-top:10px;font-size:12px;line-height:1.6;opacity:.9;}
  main{flex:1;display:flex;flex-direction:column;}
  #chat{flex:1;overflow:auto;padding:20px;}
  .msg{max-width:760px;margin:0 auto 14px;padding:12px 14px;border-radius:10px;line-height:1.7;font-size:14px;white-space:pre-wrap;}
  .user{background:#d8f0e0;margin-left:auto;}
  .ai{background:#fff;border:1px solid #e2e6e2;}
  .src{margin-top:10px;padding-top:8px;border-top:1px dashed #ccc;font-size:12px;color:#666;}
  .snip{color:#999;margin:4px 0 8px;}
  .inputbar{display:flex;padding:14px;background:#fff;border-top:1px solid #e5e5e5;}
  .inputbar input{flex:1;padding:12px;border:1px solid #ccc;border-radius:8px;font-size:14px;}
  .inputbar button{margin-left:10px;padding:0 22px;background:#3fa66a;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:14px;}
  .tip{text-align:center;color:#999;font-size:13px;margin-top:40px;}
</style>
</head>
<body>
<div class="wrap">
  <aside>
    <h2 style="font-size:18px;margin:0 0 4px;">&#127794; 林草智答</h2>
    <div style="font-size:12px;opacity:.7;">基于规程知识库的智能问答</div>
    <h3>&#9313; 模型设置</h3>
    <input id="api_key" type="password" placeholder="大模型 API Key">
    <input id="base_url" value="https://dashscope.aliyuncs.com/compatible-mode/v1">
    <input id="chat_model" value="qwen-plus">
    <input id="emb_model" value="text-embedding-v3">
    <h3>&#9314; 上传知识库文档</h3>
    <input id="files" type="file" multiple accept=".docx,.txt,.md">
    <button onclick="buildKB()">构建知识库</button>
    <div id="status"></div>
  </aside>
  <main>
    <div id="chat"><div class="tip">先在左侧填写 API Key、上传文档并构建知识库，然后开始提问。</div></div>
    <div class="inputbar">
      <input id="q" placeholder="输入问题，例如：古树群如何认定？" onkeydown="if(event.key==='Enter')ask()">
      <button onclick="ask()">发送</button>
    </div>
  </main>
</div>
<script>
function el(id){return document.getElementById(id);}
async function buildKB(){
  var fd=new FormData();
  fd.append("api_key",el("api_key").value);
  fd.append("base_url",el("base_url").value);
  fd.append("chat_model",el("chat_model").value);
  fd.append("emb_model",el("emb_model").value);
  var fs=el("files").files;
  if(!fs.length){el("status").textContent="请先选择文档";return;}
  for(var i=0;i<fs.length;i++){fd.append("files",fs[i]);}
  el("status").textContent="正在读取并构建知识库，请稍候...";
  try{
    var r=await fetch("/build",{method:"POST",body:fd});
    var j=await r.json();
    el("status").textContent=j.msg;
  }catch(e){el("status").textContent="构建失败："+e;}
}
function addMsg(role,html){
  var chat=el("chat");
  var tip=chat.querySelector(".tip");if(tip){tip.remove();}
  var d=document.createElement("div");
  d.className="msg "+role;d.innerHTML=html;
  chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d;
}
async function ask(){
  var q=el("q").value.trim();if(!q){return;}
  addMsg("user",q);el("q").value="";
  var d=addMsg("ai","检索中...");
  try{
    var r=await fetch("/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({q:q})});
    var j=await r.json();
    var html=j.answer||j.msg||"出错了";
    if(j.sources&&j.sources.length){
      html+='<div class="src"><b>引用来源</b>';
      for(var i=0;i<j.sources.length;i++){
        var s=j.sources[i];
        html+='<div>'+s.source+'（相关度 '+s.score+'）<div class="snip">'+s.snippet+'...</div></div>';
      }
      html+='</div>';
    }
    d.innerHTML=html;
  }catch(e){d.innerHTML="请求失败："+e;}
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("启动成功后，请在浏览器打开： http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
