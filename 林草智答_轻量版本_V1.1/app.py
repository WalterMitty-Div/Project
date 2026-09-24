# -*- coding: utf-8 -*-
"""林草智答 · v1.1
- 三个页面：聊天 / 知识库 / 设置
- 设置页可切换「云端大模型 / 本地小模型」
- 仅依赖 Flask，其余标准库实现
"""
import io
import json
import math
import zipfile
import urllib.request
from xml.etree import ElementTree as ET

from flask import Flask, request, jsonify, Response

app = Flask(__name__)

PRESETS = {
    "cloud": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_model": "qwen-plus",
        "emb_model": "text-embedding-v3",
    },
    "local": {
        "base_url": "http://127.0.0.1:11434/v1",
        "chat_model": "qwen2.5:7b",
        "emb_model": "bge-m3",
    },
}

STATE = {"chunks": [], "embs": [], "kb_info": "尚未构建知识库"}
HANDOFF_WORDS = ["投诉", "人工", "转客服", "举报", "找领导"]
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


# ---------------- 配置与调用 ----------------
def parse_config(raw):
    try:
        cfg = json.loads(raw) if raw else {}
    except Exception:
        cfg = {}
    mode = cfg.get("mode", "cloud")
    preset = PRESETS.get(mode, PRESETS["cloud"])
    return {
        "mode": mode,
        "api_key": (cfg.get("api_key") or "").strip(),
        "base_url": (cfg.get("base_url") or "").strip() or preset["base_url"],
        "chat_model": (cfg.get("chat_model") or "").strip() or preset["chat_model"],
        "emb_model": (cfg.get("emb_model") or "").strip() or preset["emb_model"],
    }


def http_post_json(url, payload, api_key):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def embed_texts(cfg, texts):
    url = cfg["base_url"].rstrip("/") + "/embeddings"
    res = http_post_json(url, {"model": cfg["emb_model"], "input": texts}, cfg["api_key"])
    return [d["embedding"] for d in res["data"]]


def chat_answer(cfg, question, contexts):
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    ctx = "\n\n".join("[资料%d]（来源：%s）\n%s" % (i + 1, c[0], c[1]) for i, c in enumerate(contexts))
    prompt = (
        "你是林草行业技术助手。请只依据下面提供的资料回答用户问题；"
        "如果资料中没有答案，请如实说明并建议转人工。"
        "回答要简洁、专业，并在末尾用【出处】标注用到的资料编号。\n\n"
        "资料：\n" + ctx + "\n\n用户问题：" + question
    )
    payload = {"model": cfg["chat_model"], "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    res = http_post_json(url, payload, cfg["api_key"])
    return res["choices"][0]["message"]["content"]


# ---------------- 文档处理 ----------------
def docx_to_text(data):
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
    cfg = parse_config(request.form.get("config"))
    files = request.files.getlist("files")
    if not files:
        return jsonify({"ok": False, "msg": "请先选择至少一份文档"})
    if cfg["mode"] == "cloud" and not cfg["api_key"]:
        return jsonify({"ok": False, "msg": "云端模式需要填写 API Key"})

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
        embs, batch = [], 10
        for i in range(0, len(chunks), batch):
            embs.extend(embed_texts(cfg, [c for _, c in chunks[i:i + batch]]))
    except Exception as e:
        return jsonify({"ok": False, "msg": "向量化失败：" + str(e)})

    STATE["chunks"] = chunks
    STATE["embs"] = embs
    STATE["kb_info"] = "已加载 %d 个片段，来自 %d 份文档（向量模型：%s）" % (len(chunks), len(files), cfg["emb_model"])
    return jsonify({"ok": True, "msg": "知识库已就绪：" + STATE["kb_info"]})


@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True, silent=True) or {}
    q = (body.get("q") or "").strip()
    cfg = parse_config(json.dumps(body.get("config") or {}))
    if not q:
        return jsonify({"ok": False, "msg": "请输入问题"})
    if not STATE["chunks"]:
        return jsonify({"ok": False, "msg": "请先在「知识库」页构建知识库"})
    if cfg["mode"] == "cloud" and not cfg["api_key"]:
        return jsonify({"ok": False, "msg": "云端模式需要填写 API Key"})
    if any(w in q for w in HANDOFF_WORDS):
        return jsonify({"ok": True, "answer": "这个问题我先为你转接人工客服，请稍候。", "sources": []})

    try:
        qv = embed_texts(cfg, [q])[0]
        scored = sorted(((cosine(qv, e), i) for i, e in enumerate(STATE["embs"])), reverse=True)
        contexts, sources = [], []
        for sc, i in top4(scored):
            src, txt = STATE["chunks"][i]
            contexts.append((src, txt))
            sources.append({"source": src, "snippet": txt[:200], "score": round(float(sc), 3)})
        ans = chat_answer(cfg, q, contexts)
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
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:"Microsoft YaHei",sans-serif;background:#eef2ee;color:#22302a;height:100vh;display:flex;flex-direction:column;}
  header{background:#1f3d2b;color:#fff;padding:0 22px;height:58px;display:flex;align-items:center;justify-content:space-between;}
  header .logo{font-size:18px;font-weight:600;letter-spacing:1px;}
  nav{display:flex;gap:6px;}
  nav button{background:transparent;color:#cfe3d6;border:none;padding:8px 18px;border-radius:8px;font-size:14px;cursor:pointer;}
  nav button:hover{background:#2c5440;}
  nav button.active{background:#3fa66a;color:#fff;}
  .badge{font-size:12px;background:#3fa66a;color:#fff;border-radius:20px;padding:3px 10px;margin-left:8px;}
  main{flex:1;overflow:hidden;display:flex;flex-direction:column;}
  .page{flex:1;overflow:auto;display:none;padding:24px;}
  .page.active{display:flex;flex-direction:column;}
  .card{background:#fff;border-radius:12px;padding:22px;box-shadow:0 1px 4px rgba(0,0,0,.06);max-width:820px;width:100%;margin:0 auto 16px;}
  .card h3{font-size:15px;margin-bottom:14px;color:#1f3d2b;}
  label{display:block;font-size:13px;color:#5a6b60;margin:10px 0 4px;}
  input[type=text],input[type=password]{width:100%;padding:9px 11px;border:1px solid #cfd8d1;border-radius:8px;font-size:14px;}
  input[type=file]{font-size:13px;}
  button.primary{background:#3fa66a;color:#fff;border:none;border-radius:8px;padding:10px 20px;font-size:14px;cursor:pointer;}
  button.primary:hover{background:#358c59;}
  .modes{display:flex;gap:12px;margin-bottom:8px;}
  .mode{flex:1;border:2px solid #d7e0d9;border-radius:10px;padding:14px;cursor:pointer;transition:.15s;}
  .mode.active{border-color:#3fa66a;background:#f0faf4;}
  .mode b{display:block;font-size:14px;margin-bottom:4px;}
  .mode span{font-size:12px;color:#7b8a80;}
  .hint{font-size:12px;color:#8a978e;margin-top:8px;line-height:1.6;}
  #kb_status,#status{margin-top:12px;font-size:13px;color:#3fa66a;line-height:1.6;}
  #chat{flex:1;overflow:auto;padding:6px 0;}
  .msg{max-width:760px;margin:0 auto 14px;padding:12px 15px;border-radius:12px;line-height:1.7;font-size:14px;white-space:pre-wrap;}
  .user{background:#d8f0e0;margin-right:6px;}
  .ai{background:#fff;border:1px solid #e2e6e2;}
  .src{margin-top:10px;padding-top:8px;border-top:1px dashed #ccc;font-size:12px;color:#666;}
  .snip{color:#9aa59d;margin:4px 0 8px;}
  .inputbar{display:flex;padding:14px 24px;background:#fff;border-top:1px solid #e5e5e5;}
  .inputbar input{flex:1;padding:12px;border:1px solid #cfd8d1;border-radius:10px;font-size:14px;}
  .inputbar button{margin-left:10px;padding:0 24px;background:#3fa66a;color:#fff;border:none;border-radius:10px;cursor:pointer;font-size:14px;}
  .tip{text-align:center;color:#9aa59d;font-size:13px;margin-top:50px;}
  .chathead{max-width:760px;margin:0 auto 10px;font-size:12px;color:#8a978e;display:flex;justify-content:space-between;}
</style>
</head>
<body>
<header>
  <div class="logo">&#127794; 林草智答</div>
  <nav>
    <button id="nav-chat" class="active" onclick="tab('chat')">聊天</button>
    <button id="nav-kb" onclick="tab('kb')">知识库</button>
    <button id="nav-set" onclick="tab('set')">设置</button>
  </nav>
  <div>当前模型：<span id="modeBadge" class="badge">云端大模型</span></div>
</header>

<main>
  <!-- 聊天页 -->
  <section id="page-chat" class="page active" style="padding:16px 0;">
    <div id="chat" style="padding:0 24px;">
      <div class="tip">先在「设置」确认模型，再到「知识库」上传文档，然后开始提问。</div>
    </div>
    <div class="inputbar">
      <input id="q" placeholder="输入问题，例如：古树群如何认定？" onkeydown="if(event.key==='Enter')ask()">
      <button onclick="ask()">发送</button>
    </div>
  </section>

  <!-- 知识库页 -->
  <section id="page-kb" class="page">
    <div class="card">
      <h3>&#128193; 构建知识库</h3>
      <input id="files" type="file" multiple accept=".docx,.txt,.md">
      <div style="margin-top:14px;"><button class="primary" onclick="buildKB()">构建知识库</button></div>
      <div id="kb_status"></div>
      <div class="hint">支持 Word、文本、Markdown。切换向量模型后需要重新构建知识库。</div>
    </div>
  </section>

  <!-- 设置页 -->
  <section id="page-set" class="page">
    <div class="card">
      <h3>&#9881; 模型设置</h3>
      <div class="modes">
        <div class="mode active" id="mode-cloud" onclick="setMode('cloud')"><b>云端大模型</b><span>效果最好，需联网、需 API Key</span></div>
        <div class="mode" id="mode-local" onclick="setMode('local')"><b>本地小模型</b><span>免费离线，适合外业无网环境</span></div>
      </div>
      <label>接口地址</label><input id="base_url" type="text">
      <label>对话模型</label><input id="chat_model" type="text">
      <label>向量模型</label><input id="emb_model" type="text">
      <label>API Key（本地模型可留空）</label><input id="api_key" type="password" placeholder="sk-...">
      <div style="margin-top:16px;"><button class="primary" onclick="saveCfg()">保存设置</button></div>
      <div id="status"></div>
      <div class="hint">本地小模型需先安装 Ollama 并执行：ollama pull qwen2.5:7b 与 ollama pull bge-m3</div>
    </div>
  </section>
</main>

<script>
var PRESETS = {
  cloud:{base_url:"https://dashscope.aliyuncs.com/compatible-mode/v1",chat_model:"qwen-plus",emb_model:"text-embedding-v3"},
  local:{base_url:"http://127.0.0.1:11434/v1",chat_model:"qwen2.5:7b",emb_model:"bge-m3"}
};
var cfg = {mode:"cloud",base_url:"",chat_model:"",emb_model:"",api_key:""};

function el(id){return document.getElementById(id);}
function loadCfg(){
  try{ var s=localStorage.getItem("llm_config"); if(s){cfg=JSON.parse(s);} }catch(e){}
  if(!cfg.base_url){ cfg=Object.assign({},PRESETS[cfg.mode||"cloud"]); cfg.mode=cfg.mode||"cloud"; }
  paintCfg();
}
function paintCfg(){
  el("base_url").value=cfg.base_url||"";
  el("chat_model").value=cfg.chat_model||"";
  el("emb_model").value=cfg.emb_model||"";
  el("api_key").value=cfg.api_key||"";
  el("mode-cloud").className = "mode"+(cfg.mode==="cloud"?" active":"");
  el("mode-local").className = "mode"+(cfg.mode==="local"?" active":"");
  el("modeBadge").textContent = cfg.mode==="local" ? "本地小模型" : "云端大模型";
}
function setMode(m){
  cfg.mode=m;
  var p=PRESETS[m];
  cfg.base_url=p.base_url; cfg.chat_model=p.chat_model; cfg.emb_model=p.emb_model;
  if(m==="local"){ cfg.api_key=""; }
  paintCfg();
}
function saveCfg(){
  cfg.base_url=el("base_url").value; cfg.chat_model=el("chat_model").value;
  cfg.emb_model=el("emb_model").value; cfg.api_key=el("api_key").value;
  localStorage.setItem("llm_config", JSON.stringify(cfg));
  el("status").textContent="设置已保存，当前使用："+(cfg.mode==="local"?"本地小模型":"云端大模型");
}
function tab(n){
  var pages=["chat","kb","set"];
  for(var i=0;i<pages.length;i++){
    el("page-"+pages[i]).className="page"+(pages[i]===n?" active":"");
    el("nav-"+pages[i]).className=(pages[i]===n?"active":"");
  }
}
function addMsg(role,html){
  var chat=el("chat");
  var tip=chat.querySelector(".tip"); if(tip){tip.remove();}
  var d=document.createElement("div"); d.className="msg "+role; d.innerHTML=html;
  chat.appendChild(d); chat.scrollTop=chat.scrollHeight; return d;
}
async function buildKB(){
  var fs=el("files").files;
  if(!fs.length){ el("kb_status").textContent="请先选择文档"; return; }
  var fd=new FormData(); fd.append("config", JSON.stringify(cfg));
  for(var i=0;i<fs.length;i++){ fd.append("files", fs[i]); }
  el("kb_status").textContent="正在读取并构建知识库，请稍候...";
  try{ var r=await fetch("/build",{method:"POST",body:fd}); var j=await r.json(); el("kb_status").textContent=j.msg; }
  catch(e){ el("kb_status").textContent="构建失败："+e; }
}
async function ask(){
  var q=el("q").value.trim(); if(!q){return;}
  addMsg("user",q); el("q").value="";
  var d=addMsg("ai","检索中...");
  try{
    var r=await fetch("/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({q:q,config:cfg})});
    var j=await r.json();
    var html=j.answer||j.msg||"出错了";
    if(j.sources&&j.sources.length){
      html+='<div class="src"><b>引用来源</b>';
      for(var i=0;i<j.sources.length;i++){ var s=j.sources[i];
        html+='<div>'+s.source+'（相关度 '+s.score+'）<div class="snip">'+s.snippet+'...</div></div>'; }
      html+='</div>';
    }
    d.innerHTML=html;
  }catch(e){ d.innerHTML="请求失败："+e; }
}
loadCfg();
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("启动成功后，请在浏览器打开： http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
