import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 🔑 1. നിങ്ങളുടെ Claude API Key ഇവിടെ നൽകുക
os.environ["ANTHROPIC_API_KEY"] = "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"

# 2. Embedding Model സെറ്റപ്പ്
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None

def add_document(file_path):
    """ പുതിയ ഡോക്യുമെന്റ് സിസ്റ്റത്തിലേക്ക് ചേർക്കാൻ """
    global vector_store
    
    print(f"📄 '{file_path}' വായിക്കുന്നു...")
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
        
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    if vector_store is None:
        vector_store = FAISS.from_documents(splits, embeddings)
    else:
        vector_store.add_documents(splits)
        
    print(f"✅ '{file_path}' വിജയകരമായി ഇൻഡക്സ് ചെയ്തു!\n")

# 3. Claude AI Prompt (മലയാളം + Personal Thinking)
CLAUDE_SYSTEM_PROMPT = """
You are an intelligent and thoughtful AI assistant.

YOUR ROLE:
Read the provided Context Documents carefully. Think through the underlying logical details step-by-step, and answer the question in clear, natural MALAYALAM (മലയാളം).

RULES:
1. Personal Thinking & Reasoning: Analyze the facts, connect related concepts, and formulate a well-thought-out response before outputting.
2. Context Bound: Rely strictly on the information given in the Context.
3. Language: Write the entire answer ONLY in polite Malayalam.
4. If missing info: Reply "ക്ഷമിക്കണം, നൽകിയിട്ടുള്ള ഡോക്യുമെന്റുകളിൽ ഈ ചോദ്യത്തിനുള്ള ഉത്തരം ലഭ്യമല്ല."

Context:
{context}

Question:
{question}

Answer in Malayalam:
"""

prompt = PromptTemplate(
    template=CLAUDE_SYSTEM_PROMPT,
    input_variables=["context", "question"]
)

# 4. LLM Model
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.3
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_bot(question):
    """ ചോദ്യങ്ങൾക്ക് മറുപടി നൽകുന്ന ഫംഗ്ഷൻ """
    if vector_store is None:
        return "ദയവായി ആദ്യം ഒരു ഡോക്യുമെന്റ് ചേർക്കുക."
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(question)

# 5. ചാറ്റ് പ്രോഗ്രാം പ്രവർത്തിപ്പിക്കുന്ന ഭാഗം
if __name__ == "__main__":
    # ⚠️ ശ്രദ്ധിക്കുക: ഈ ഫോൾഡറിലുള്ള നിങ്ങളുടെ PDF ഫയലിന്റെ പേര് താഴെ നൽകുക
    doc_name = "sample.pdf" 
    
    if os.path.exists(doc_name):
        add_document(doc_name)
        
        print("🤖 മലയാളം ചാറ്റ്‌ബോട്ട് തയ്യാറാണ്! ('exit' എന്ന് ടൈപ്പ് ചെയ്താൽ പുറത്തുകടക്കാം)\n")
        
        # തുടർച്ചയായി ചോദ്യങ്ങൾ ചോദിക്കാനുള്ള ചാറ്റ് ലൂപ്പ്
        while True:
            user_input = input("നിങ്ങളുടെ ചോദ്യം ചോദിക്കുക: ")
            if user_input.lower() == 'exit':
                print("നന്ദി!")
                break
                
            if user_input.strip() != "":
                print("ചിന്തിക്കുന്നു... ⏳")
                answer = ask_bot(user_input)
                print(f"\nAI മറുപടി:\n{answer}\n")
                print("-" * 50)
    else:
        print(f"❌ '{doc_name}' എന്ന ഫയൽ കണ്ടുപിടിക്കാൻ കഴിഞ്ഞില്ല. ദയവായി ഈ ഫോൾഡറിലേക്ക് ഫയൽ കോപ്പി ചെയ്തിടുക.")