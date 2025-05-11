import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import tempfile
import os
import gdown
import logging

# Configure minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(page_title="Tap Bonds AI Chatbot", page_icon="📊")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "data_status" not in st.session_state:
    st.session_state.data_status = {
        "bond": {"status": "not_started", "message": "Not loaded"},
        "cashflow": {"status": "not_started", "message": "Not loaded"},
        "company": {"status": "not_started", "message": "Not loaded"}
    }

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_file_id_from_url(url):
    """Extract file ID from Google Drive URL"""
    if not url:
        return None
    try:
        if '/file/d/' in url:
            return url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            return url.split('id=')[1].split('&')[0]
        elif '/open?id=' in url:
            return url.split('/open?id=')[1].split('&')[0]
        else:
            return url.split('/')[-2]
    except Exception:
        return None

def load_csv_from_drive(url, doc_type):
    """Load CSV from Google Drive and return documents using CSVLoader"""
    file_id = get_file_id_from_url(url)
    if not file_id:
        return None, f"Invalid URL: {url}"
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            temp_path = temp_file.name
        download_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(download_url, temp_path, quiet=True)
        
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            loader = CSVLoader(
                file_path=temp_path,
                csv_args={"delimiter": ","},
                metadata_columns=["isin", "company_name"] if doc_type != "company" else ["company_name"]
            )
            documents = loader.load()
            os.unlink(temp_path)
            
            if not documents:
                return None, "CSV file is empty"
            
            # Add doc_type to metadata
            for doc in documents:
                doc.metadata["doc_type"] = doc_type
            
            # Validate data
            if doc_type == "bond":
                coupon_rates = [float(doc.page_content.split("coupon_rate:")[1].split("\n")[0]) 
                               for doc in documents if "coupon_rate:" in doc.page_content]
                if coupon_rates and all(rate == 0 for rate in coupon_rates):
                    st.warning("Warning: All coupon rates are 0. Filter queries may return no results.")
                security_types = [doc.page_content.split("security_type:")[1].split("\n")[0].lower() 
                                 for doc in documents if "security_type:" in doc.page_content]
                if security_types and all(st == "n/a" for st in security_types):
                    st.warning("Warning: All security types are 'N/A'. Secured bond queries may fail.")
            
            return documents, f"Loaded {len(documents)} {doc_type} records"
        return None, "Downloaded file is empty"
    except Exception as e:
        try:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        except:
            pass
        return None, f"Error loading file: {str(e)}"

def load_all_data(bond_urls, cashflow_url, company_url):
    """Load bond, cashflow, and company data"""
    status = {
        "bond": {"status": "not_started", "message": "Not loaded"},
        "cashflow": {"status": "not_started", "message": "Not loaded"},
        "company": {"status": "not_started", "message": "Not loaded"}
    }
    all_documents = []
    
    # Load bond files
    if bond_urls and any(bond_urls):
        status["bond"]["status"] = "in_progress"
        for i, url in enumerate(bond_urls):
            if not url:
                continue
            docs, message = load_csv_from_drive(url, "bond")
            if docs:
                all_documents.extend(docs)
                status["bond"]["status"] = "success"
                status["bond"]["message"] = message
            else:
                status["bond"]["status"] = "error"
                status["bond"]["message"] = f"Bond file {i+1}: {message}"
    
    # Load cashflow file
    if cashflow_url:
        status["cashflow"]["status"] = "in_progress"
        docs, message = load_csv_from_drive(cashflow_url, "cashflow")
        if docs:
            all_documents.extend(docs)
            status["cashflow"]["status"] = "success"
            status["cashflow"]["message"] = message
        else:
            status["cashflow"]["status"] = "error"
            status["cashflow"]["message"] = message
    
    # Load company file
    if company_url:
        status["company"]["status"] = "in_progress"
        docs, message = load_csv_from_drive(company_url, "company")
        if docs:
            all_documents.extend(docs)
            status["company"]["status"] = "success"
            status["company"]["message"] = message
        else:
            status["company"]["status"] = "error"
            status["company"]["message"] = message
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    split_documents = text_splitter.split_documents(all_documents)
    
    # Create vector store
    if split_documents:
        vector_store = FAISS.from_documents(split_documents, embeddings)
        st.session_state.vector_store = vector_store
        return status
    return status

def get_llm(api_key, model_option, temperature, max_tokens):
    """Initialize ChatGroq model"""
    if not api_key:
        return None
    try:
        return ChatGroq(
            model=model_option,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception:
        return None

def create_rag_chain(vector_store, llm):
    """Create RAG prompt and chain"""
    prompt_template = """
    You are a financial assistant specializing in bonds. Use the provided context to answer the query in a professional, friendly manner with Markdown formatting. Structure responses as tables for lists of bonds or cashflows, and use text for single bond details or company metrics. If the context is insufficient or data is missing (e.g., coupon_rate is 0 or security_type is N/A), provide a clear error message suggesting the user check the CSV files.

    **User Query**: {question}
    **Context**: {context}

    **Instructions**:
    - For ISIN lookups (e.g., "Show details for INE123456789"), return a single bond's details in a formatted text block.
    - For company issuances (e.g., "Show all issuances by Ugro Capital"), return a table of bonds.
    - For filter searches (e.g., "Find secured debentures with a coupon rate above 5%"), return a table of matching bonds, filtering by coupon_rate and security_type.
    - For cashflow queries (e.g., "Cash flow for INE123456789"), return a table of cashflow dates and types.
    - For company metrics (e.g., "What is the EPS of ABC company?"), return the requested metric (e.g., EPS, rating).
    - For bond finder queries (e.g., "List bonds with a yield of more than 9%"), return a table of bonds with yields above the threshold (use mock data if needed).
    - If no data matches, return: "**Error**: No results found. Please check if the CSV files contain valid coupon_rate or security_type values."
    - Always format tables with columns aligned and include up to 5 rows for brevity.

    **Output Format**:
    - Use Markdown tables for lists (e.g., | ISIN | Issuer | Coupon Rate | ... |).
    - Use bullet points or text for single records.
    - Include a header like "## Response" for clarity.
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"]
    )
    
    def format_context(documents):
        """Format retrieved documents into context string"""
        context = ""
        for doc in documents:
            context += f"Document Type: {doc.metadata.get('doc_type', 'unknown')}\n"
            context += f"Metadata: {doc.metadata}\n"
            context += f"Content: {doc.page_content}\n\n"
        return context
    
    def rag_chain(query):
        """Run RAG pipeline"""
        # Retrieve relevant documents
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 50}
        )
        docs = retriever.invoke(query)
        
        # Filter by metadata if ISIN or company_name is in query
        if "INE" in query.upper():
            isin = next((word for word in query.split() if word.upper().startswith("INE") and len(word) >= 10), None)
            if isin:
                docs = [doc for doc in docs if doc.metadata.get("isin") == isin]
        elif "by" in query.lower():
            company_name = query.lower().split("by")[-1].strip()
            docs = [doc for doc in docs if company_name.lower() in doc.metadata.get("company_name", "").lower()]
        
        # Format context
        context = format_context(docs)
        
        # Generate response
        response = llm.invoke(prompt.format(question=query, context=context)).content
        return response
    
    return rag_chain

def display_status_indicator(status):
    """Display status icons"""
    if status == "success":
        return "[SUCCESS]"
    elif status == "error":
        return "[ERROR]"
    elif status == "in_progress":
        return "[IN PROGRESS]"
    else:
        return "[NOT STARTED]"

def main():
    """Main application function"""
    # Sidebar configuration
    with st.sidebar:
        st.title("Configuration")
        api_key = st.text_input("Enter GROQ API Key", type="password")
        
        st.markdown("### Google Drive Integration")
        st.markdown("#### Bond Detail Files URLs")
        bond_urls = []
        for i in range(4):
            bond_url = st.text_input(f"Bond Details CSV Part {i+1} URL", key=f"bond_url_{i}")
            bond_urls.append(bond_url)
        
        cashflow_url = st.text_input("Cashflow Details CSV URL")
        company_url = st.text_input("Company Insights CSV URL")
        
        if st.button("Load Data from Drive"):
            with st.spinner("Loading data..."):
                st.session_state.data_status = load_all_data(bond_urls, cashflow_url, company_url)
                if st.session_state.vector_store:
                    st.success("Data loaded successfully!")
                else:
                    st.error("Failed to load data. Check Debug Information.")
        
        st.markdown("### Model Configuration")
        model_option = st.selectbox("Select Model", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"])
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        max_tokens = st.slider("Max Tokens", min_value=500, max_value=4000, value=1500, step=500)
    
    # Main content
    st.title("Tap Bonds AI Chatbot")
    st.markdown("""
    Welcome to the Tap Bonds AI Chatbot! 💼🔍

    Ask about bonds, companies, cash flows, yields, or search for bonds.

    **Example queries:**
    - "Show details for INE123456789"
    - "Show all issuances by Ugro Capital"
    - "Find secured debentures with a coupon rate above 10%"
    - "What is the EPS of ABC company?"
    - "List bonds with a yield of more than 9%"
    """)

    # Data status
    st.markdown("### Data Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        bond_status = st.session_state.data_status.get("bond", {"status": "not_started", "message": "Not loaded"})
        st.markdown(f"{display_status_indicator(bond_status['status'])} **Bond Data:** {bond_status['message']}")
    with col2:
        cashflow_status = st.session_state.data_status.get("cashflow", {"status": "not_started", "message": "Not loaded"})
        st.markdown(f"{display_status_indicator(cashflow_status['status'])} **Cashflow Data:** {cashflow_status['message']}")
    with col3:
        company_status = st.session_state.data_status.get("company", {"status": "not_started", "message": "Not loaded"})
        st.markdown(f"{display_status_indicator(company_status['status'])} **Company Data:** {company_status['message']}")

    if not api_key:
        st.warning("⚠️ Please enter your GROQ API key in the sidebar.")

    st.markdown("---")

    # Query input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Enter your query:", key="query_input")
    with col2:
        submit_button = st.button("Submit", use_container_width=True)

    # Process query
    if submit_button and query:
        llm = get_llm(api_key, model_option, temperature, max_tokens)
        if llm and st.session_state.vector_store:
            rag_chain = create_rag_chain(st.session_state.vector_store, llm)
            with st.spinner("Processing your query..."):
                st.session_state.chat_history.append({"role": "user", "content": query})
                try:
                    response = rag_chain(query)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    response = f"**Error**: Failed to process query: {str(e)}"
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.error("Please enter a valid GROQ API key and load data.")

    # Display chat history
    st.markdown("### Conversation")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You**: {message['content']}")
        else:
            st.markdown(f"**Tap Bonds AI**: {message['content']}")

    # Footer
    st.markdown("---")
    st.markdown("Powered by Tap Bonds AI")

    # Debug information
    with st.expander("Debug Information", expanded=False):
        st.write("Data Availability:")
        for key, status in st.session_state.data_status.items():
            st.write(f"- {key.capitalize()} Data: {display_status_indicator(status['status'])} {status['message']}")
        
        if st.checkbox("Show Vector Store Stats"):
            if st.session_state.vector_store:
                st.write(f"Vector Store: {len(st.session_state.vector_store.index_to_docstore_id)} documents indexed")

if __name__ == "__main__":
    main()
