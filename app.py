import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="മലയാളം AI ഡോക്യുമെന്റ് അസിസ്റ്റന്റ്", page_icon="🤖", layout="wide")

st.title("🤖 മലയാളം AI ഡോക്യുമെന്റ് അസിസ്റ്റന്റ്")
st.write("നൽകിയിരിക്കുന്ന PDF/TXT ഡോക്യുമെന്റുകളിൽ നിന്ന് ചിന്തിച്ച് മലയാളത്തിൽ മറുപടി നൽകുന്ന ബോട്ട്.")

# =========================================================
# 🔑 1. API Key & File Settings (Sidebar)
# =========================================================
st.sidebar.header("⚙️ ക്രമീകരണങ്ങൾ (Settings)")

default_api_key = "YOUR_NEW_API_KEY_HERE"  # 👈 നിങ്ങളുടെ API Key ഇവിടെ നൽകുക
api_key = st.sidebar.text_input("Claude API Key:", value=default_api_key, type="password")

default_pdf_path = "All pdf.pdf"

# =========================================================
# 🧠 2. Core Bot Logic Function (PDF & TXT Support)
# =========================================================
@st.cache_resource
def load_and_process_file(file_path):
    """ PDF അല്ലെങ്കിൽ TXT ഫയലുകൾ വായിച്ച് Vector DB ഉണ്ടാക്കുന്നു """
    embeddings = FastEmbedEmbeddings()
    
    # ഫയൽ ടൈപ്പ് പരിശോധിക്കുന്നു
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        # .txt ഫയലുകൾ വായിക്കാൻ
        loader = TextLoader(file_path, encoding='utf-8')
        
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

# =========================================================
# 🚀 3. App Execution logic
# =========================================================
if api_key and api_key != "YOUR_NEW_API_KEY_HERE":
    os.environ["ANTHROPIC_API_KEY"] = api_key
    
    st.sidebar.subheader("📄 ഡോക്യുമെന്റ് തിരഞ്ഞെടുക്കുക")
    
    # 💡 ഇവിടെ .pdf, .txt ഫയലുകൾ അപ്‌ലോഡ് ചെയ്യാനുള്ള അനുമതി നൽകിയിരിക്കുന്നു
    uploaded_file = st.sidebar.file_uploader("പുതിയ PDF/TXT ഫയൽ ചേർക്കുക", type=["pdf", "txt"])
    
    active_file = None
    
    # 1. ഉപയോക്താവ് ഫയൽ അപ്‌ലോഡ് ചെയ്താൽ
    if uploaded_file is not None:
        file_extension = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
        active_file = f"temp_uploaded{file_extension}"
        
        with open(active_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"പുതിയ ഫയൽ ലോഡ് ചെയ്തു: {uploaded_file.name}")
        
    # 2. അല്ലങ്കിൽ ഡെഫോൾട്ട് 'All pdf.pdf' ഫയൽ
    elif os.path.exists(default_pdf_path):
        active_file = default_pdf_path
        st.sidebar.info(f"പ്രധാന ഫയൽ ലോഡ് ചെയ്തിരിക്കുന്നു: {default_pdf_path}")
    else:
        st.error(f"❌ '{default_pdf_path}' എന്ന ഫയൽ കണ്ടെത്താനായില്ല. ഒരു PDF/TXT ഫയൽ അപ്‌ലോഡ് ചെയ്യുക.")

    # ഫയൽ പ്രോസസ്സ് ചെയ്യുന്നു
    if active_file:
        with st.spinner("ഡോക്യുമെന്റ് പ്രോസസ്സ് ചെയ്യുന്നു, ദയവായി കാത്തിരിക്കുക... ⏳"):
            try:
                vector_store = load_and_process_file(active_file)
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                
                CLAUDE_SYSTEM_PROMPT = """
                You are an intelligent, empathetic, and highly analytical AI assistant.

                YOUR GOAL:
                Answer the user's question accurately in MALAYALAM (മലയാളം) based on the provided Context Documents.

                INSTRUCTIONS & PERSONAL THINKING:
                1. **Analyze & Reason (Personal Thinking):** Analyze the provided context and the user's question carefully step-by-step.
                2. **Strict Context Alignment:** Rely strictly on the information given in the Context Documents. Do not invent facts.
                3. **Malayalam Output:** Write the final answer ONLY in clear, natural, and polite Malayalam.
                4. **Handling Missing Info:** If the context does not contain enough information, state politely:
                   "ക്ഷമിക്കണം, നൽകിയിട്ടുള്ള ഡോക്യുമെന്റുകളിൽ ഈ ചോദ്യത്തിനുള്ള ഉത്തരം ലഭ്യമല്ല."

                Context Documents:
                {context}

                Question:
                {question}

                Answer in Malayalam:
                """
                
                prompt = PromptTemplate(template=CLAUDE_SYSTEM_PROMPT, input_variables=["context", "question"])
                llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                st.success("✅ ചാറ്റ്‌ബോട്ട് ചോദിക്കാൻ തയ്യാറാണ്!")
                
                st.write("---")
                user_question = st.text_input("നിങ്ങളുടെ ചോദ്യം മലയാളത്തിൽ ചോദിക്കുക:")
                
                if user_question:
                    with st.spinner("ഉത്തരം ചിന്തിച്ച് തയ്യാറാക്കുന്നു... ⏳"):
                        response = rag_chain.invoke(user_question)
                        st.subheader("💡 AI മറുപടി:")
                        st.write(response)

            except Exception as e:
                st.error(f"പ്രോസസ്സിംഗിൽ തടസ്സം നേരിട്ടു: {e}")

else:
    st.warning("⚠️ തുടരുന്നതിനായി ദയവായി സൈഡ്ബാറിൽ നിങ്ങളുടെ Claude API Key നൽകുക.")
