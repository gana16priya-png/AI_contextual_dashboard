import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ──────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 10px 40px rgba(102,126,234,0.25);
}
.hero-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.hero-header p  { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 1rem; }

/* ── Stat cards ── */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.stat-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e8ecf1;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: transform 0.2s ease;
}
.stat-card:hover { transform: translateY(-3px); }
.stat-card .icon  { font-size: 1.6rem; }
.stat-card .value { font-size: 1.4rem; font-weight: 700; color: #1a1a2e; }
.stat-card .label { font-size: 0.78rem; color: #6c757d; margin-top: 2px; }

/* ── Chat bubbles ── */
.chat-container {
    max-height: 520px;
    overflow-y: auto;
    padding: 1rem 0.5rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}
.chat-msg {
    display: flex;
    gap: 0.75rem;
    max-width: 85%;
    animation: fadeIn 0.3s ease;
}
.chat-msg.user  { align-self: flex-end; flex-direction: row-reverse; }
.chat-msg.assistant { align-self: flex-start; }
.chat-avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.chat-msg.user .chat-avatar      { background: #667eea; color: #fff; }
.chat-msg.assistant .chat-avatar { background: #f0f0f5; color: #764ba2; }
.chat-bubble {
    padding: 0.85rem 1.15rem;
    border-radius: 16px;
    font-size: 0.92rem;
    line-height: 1.55;
    white-space: pre-wrap;
}
.chat-msg.user .chat-bubble {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    border-bottom-right-radius: 4px;
}
.chat-msg.assistant .chat-bubble {
    background: #f4f4f9;
    color: #1a1a2e;
    border: 1px solid #e8ecf1;
    border-bottom-left-radius: 4px;
}

/* ── Recommendation card ── */
.reco-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #eef1f5 100%);
    border-left: 5px solid #667eea;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    font-size: 0.92rem;
    line-height: 1.65;
    white-space: pre-wrap;
}

/* ── Upload area ── */
.upload-zone {
    border: 2px dashed rgba(255,255,255,0.25);
    border-radius: 14px;
    padding: 1.5rem 1rem;
    text-align: center;
    margin-bottom: 1rem;
}

/* ── Animations ── */
@keyframes fadeIn { from {opacity:0;transform:translateY(6px)} to {opacity:1;transform:translateY(0)} }

/* ── Scrollbar ── */
.chat-container::-webkit-scrollbar { width: 6px; }
.chat-container::-webkit-scrollbar-thumb { background: #c4c4c4; border-radius: 3px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.4,
        convert_system_message_to_human=True,
    )


def extract_pdf_text(pdf_file) -> str:
    """Extract all text from an uploaded PDF."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def build_vector_store(text: str):
    """Split text into chunks and build a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store, chunks


def get_conversation_chain(vector_store):
    """Create a ConversationalRetrievalChain with memory."""
    llm = get_llm()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an expert AI Resume Analyst. Use the resume context below to answer the user's question accurately and concisely.
If the answer is not found in the resume, say so honestly.

Resume Context:
{context}

Question: {question}

Answer:""",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=False,
    )
    return chain


def analyze_resume_vs_job(resume_text: str, job_description: str) -> str:
    """Use Gemini to compare resume against a job description."""
    llm = get_llm()

    prompt = f"""You are a senior technical recruiter and career coach. Analyze the candidate's resume against the provided job description.

Provide your analysis in the following structured format:

## ✅ Matching Skills & Keywords
List every skill, technology, or keyword from the job description that IS present in the resume.

## ❌ Missing Skills & Keywords
List every skill, technology, or keyword from the job description that is NOT found in the resume. This is the most critical section.

## 📊 Match Score
Give a percentage match score (0-100%) with a brief justification.

## 🎯 Recommendations
Provide 3-5 specific, actionable recommendations the candidate should follow to improve their resume for this role.

## 📝 Suggested Resume Bullet Points
Write 2-3 new bullet points the candidate could add to better align with the job description.

---
RESUME:
{resume_text}

---
JOB DESCRIPTION:
{job_description}
"""
    response = llm.invoke(prompt)
    return response.content


# ──────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────
defaults = {
    "vector_store": None,
    "chain": None,
    "chat_history": [],
    "resume_text": "",
    "resume_chunks": [],
    "resume_name": "",
    "recommendation": "",
    "processing": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center;padding:1.2rem 0 0.5rem 0;'>
            <span style='font-size:2.5rem;'>📄</span>
            <h2 style='margin:0.3rem 0 0 0;font-weight:700;'>Resume Agent</h2>
            <p style='font-size:0.82rem;opacity:0.7;margin-top:2px;'>Powered by Gemini 1.5 Flash</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── API Key ──
    api_key_input = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        value=GOOGLE_API_KEY or "",
        help="Get your free key at https://aistudio.google.com/app/apikey",
    )
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
        GOOGLE_API_KEY = api_key_input

    st.markdown("---")

    # ── File Upload ──
    st.markdown(
        "<div class='upload-zone'>📎 <strong>Upload Resume</strong><br><span style='font-size:0.78rem;opacity:0.65;'>PDF format • Max 200 MB</span></div>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file and uploaded_file.name != st.session_state.resume_name:
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
            st.error("⚠️ Please enter a valid Gemini API key first.")
        else:
            with st.spinner("🔍 Processing resume…"):
                try:
                    raw_text = extract_pdf_text(uploaded_file)
                    if not raw_text.strip():
                        st.error("Could not extract text. Is this a scanned PDF?")
                    else:
                        vs, chunks = build_vector_store(raw_text)
                        chain = get_conversation_chain(vs)

                        st.session_state.vector_store = vs
                        st.session_state.chain = chain
                        st.session_state.resume_text = raw_text
                        st.session_state.resume_chunks = chunks
                        st.session_state.resume_name = uploaded_file.name
                        st.session_state.chat_history = []
                        st.session_state.recommendation = ""

                        st.success(f"✅ **{uploaded_file.name}** processed!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

    # ── Resume Stats ──
    if st.session_state.resume_text:
        st.markdown("---")
        st.markdown("##### 📊 Resume Stats")
        word_count = len(st.session_state.resume_text.split())
        char_count = len(st.session_state.resume_text)
        chunk_count = len(st.session_state.resume_chunks)
        col1, col2 = st.columns(2)
        col1.metric("Words", f"{word_count:,}")
        col2.metric("Chunks", chunk_count)
        st.caption(f"📄 {st.session_state.resume_name} • {char_count:,} chars")

    # ── Quick Questions ──
    if st.session_state.chain:
        st.markdown("---")
        st.markdown("##### ⚡ Quick Questions")
        quick_qs = [
            "Summarize the resume",
            "List technical skills",
            "What is the work experience?",
            "What is the education background?",
            "List key achievements",
        ]
        for q in quick_qs:
            if st.button(q, key=f"qq_{q}", use_container_width=True):
                st.session_state.pending_question = q

    # ── Reset ──
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.recommendation = ""
        st.rerun()

    if st.button("🔄 Reset Everything", use_container_width=True):
        for k in defaults:
            st.session_state[k] = defaults[k]
        if "pending_question" in st.session_state:
            del st.session_state["pending_question"]
        st.rerun()


# ──────────────────────────────────────────────
# MAIN AREA
# ──────────────────────────────────────────────

# ── Hero Header ──
st.markdown(
    """
    <div class="hero-header">
        <h1>📄 AI Resume Agent</h1>
        <p>Upload your resume and chat with it — get insights, analysis, and job-fit recommendations instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.chain:
    # ── Landing State ──
    st.markdown(
        """
        <div class="stat-row">
            <div class="stat-card">
                <div class="icon">📤</div>
                <div class="value">1</div>
                <div class="label">Upload Resume</div>
            </div>
            <div class="stat-card">
                <div class="icon">🤖</div>
                <div class="value">2</div>
                <div class="label">Ask Questions</div>
            </div>
            <div class="stat-card">
                <div class="icon">🎯</div>
                <div class="value">3</div>
                <div class="label">Get Recommendations</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("👈 **Upload a PDF resume** from the sidebar to get started.")
    st.stop()

# ── Stat Row ──
word_count = len(st.session_state.resume_text.split())
chunk_count = len(st.session_state.resume_chunks)
chat_count = len([m for m in st.session_state.chat_history if m["role"] == "user"])
st.markdown(
    f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="icon">📄</div>
            <div class="value">{st.session_state.resume_name}</div>
            <div class="label">Active Resume</div>
        </div>
        <div class="stat-card">
            <div class="icon">📝</div>
            <div class="value">{word_count:,}</div>
            <div class="label">Words Extracted</div>
        </div>
        <div class="stat-card">
            <div class="icon">🧩</div>
            <div class="value">{chunk_count}</div>
            <div class="label">Vector Chunks</div>
        </div>
        <div class="stat-card">
            <div class="icon">💬</div>
            <div class="value">{chat_count}</div>
            <div class="label">Questions Asked</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ──
tab_chat, tab_reco = st.tabs(["💬 Chat with Resume", "🎯 Job Recommendation"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — CHAT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_chat:
    # Render chat history
    chat_html = '<div class="chat-container">'
    if not st.session_state.chat_history:
        chat_html += """
        <div style="text-align:center;padding:3rem 1rem;opacity:0.5;">
            <span style="font-size:3rem;">💬</span>
            <p style="margin-top:0.5rem;">Ask anything about the resume…</p>
        </div>
        """
    else:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            avatar = "👤" if role == "user" else "🤖"
            # Escape basic HTML in content
            content = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            chat_html += f"""
            <div class="chat-msg {role}">
                <div class="chat-avatar">{avatar}</div>
                <div class="chat-bubble">{content}</div>
            </div>
            """
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # Handle pending quick question
    pending_q = st.session_state.pop("pending_question", None)

    # Chat input
    user_input = st.chat_input("Ask a question about the resume…")
    question = pending_q or user_input

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("Thinking…"):
            try:
                response = st.session_state.chain.invoke({"question": question})
                answer = response.get("answer", "Sorry, I couldn't generate a response.")
            except Exception as e:
                answer = f"⚠️ Error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — RECOMMENDATION AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_reco:
    st.markdown("#### 🎯 Resume vs Job Description Analyzer")
    st.markdown(
        "Paste a job description below and click **Analyze** to see how well the resume matches, "
        "which keywords are missing, and get actionable recommendations."
    )

    job_desc = st.text_area(
        "Job Description",
        height=220,
        placeholder="Paste the full job description here…",
    )

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze_btn:
        if not job_desc.strip():
            st.warning("Please paste a job description first.")
        else:
            with st.spinner("🔍 Analyzing resume against job description…"):
                try:
                    result = analyze_resume_vs_job(st.session_state.resume_text, job_desc)
                    st.session_state.recommendation = result
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    if st.session_state.recommendation:
        st.markdown("---")
        st.markdown(st.session_state.recommendation)
