import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.output_parser import StrOutputParser
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import json
import tempfile
import os
import time
import logging
import requests
from bs4 import BeautifulSoup
import gdown

# Configure minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(page_title="Tap Bonds AI Chatbot", page_icon="📊")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bond_details" not in st.session_state:
    st.session_state.bond_details = None
if "cashflow_details" not in st.session_state:
    st.session_state.cashflow_details = None
if "company_insights" not in st.session_state:
    st.session_state.company_insights = None
if "data_loading_status" not in st.session_state:
    st.session_state.data_loading_status = {
        "bond": {"status": "not_started", "message": "Not loaded"},
        "cashflow": {"status": "not_started", "message": "Not loaded"},
        "company": {"status": "not_started", "message": "Not loaded"}
    }
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

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

def load_csv_from_drive_url(url):
    """Load CSV from Google Drive URL"""
    file_id = get_file_id_from_url(url)
    if not file_id:
        return None, f"Invalid URL format: {url}"
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            temp_path = temp_file.name
        download_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(download_url, temp_path, quiet=False)
        
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            df = pd.read_csv(temp_path)
            os.unlink(temp_path)
            if df.empty:
                return None, "Downloaded file is empty"
            df.columns = [col.lower() for col in df.columns]
            return df, "Success"
        return None, "Downloaded file is empty or missing"
    except Exception as e:
        try:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        except:
            pass
        return None, f"Error downloading file: {str(e)}"

def validate_csv_file(df, expected_columns):
    """Check if DataFrame has required columns"""
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    expected_columns_lower = [col.lower() for col in expected_columns]
    df_columns_lower = [col.lower() for col in df.columns]
    missing_columns = [col for col in expected_columns_lower if col not in df_columns_lower]
    if missing_columns:
        return False, f"Missing columns: {', '.join(missing_columns)}"
    return True, "DataFrame validated successfully"

def load_data_from_drive(bond_urls, cashflow_url, company_url):
    """Load data from Google Drive URLs"""
    bond_details, cashflow_details, company_insights = None, None, None
    status = {
        "bond": {"status": "not_started", "message": "Not loaded"},
        "cashflow": {"status": "not_started", "message": "Not loaded"},
        "company": {"status": "not_started", "message": "Not loaded"}
    }
    
    # Load bond files
    if bond_urls and any(bond_urls):
        status["bond"]["status"] = "in_progress"
        bond_dfs = []
        for i, url in enumerate(bond_urls):
            if not url:
                continue
            df, message = load_csv_from_drive_url(url)
            if df is not None:
                is_valid, validation_message = validate_csv_file(df, ['isin', 'company_name'])
                if is_valid:
                    bond_dfs.append(df)
                else:
                    status["bond"]["status"] = "error"
                    status["bond"]["message"] = f"Bond file {i+1}: {validation_message}"
            else:
                status["bond"]["status"] = "error"
                status["bond"]["message"] = f"Error reading bond file {i+1}: {message}"
        
        if bond_dfs:
            bond_details = pd.concat(bond_dfs, ignore_index=True)
            bond_details = bond_details.drop_duplicates(subset=['isin'], keep='first')
            status["bond"]["status"] = "success"
            status["bond"]["message"] = f"Loaded {len(bond_details)} bonds"
        elif status["bond"]["status"] != "error":
            status["bond"]["status"] = "error"
            status["bond"]["message"] = "No valid bond files processed"
    
    # Load cashflow data
    if cashflow_url:
        status["cashflow"]["status"] = "in_progress"
        cashflow_details, message = load_csv_from_drive_url(cashflow_url)
        if cashflow_details is not None:
            is_valid, validation_message = validate_csv_file(cashflow_details, ['isin', 'cash_flow_date', 'cash_flow_amount'])
            if is_valid:
                status["cashflow"]["status"] = "success"
                status["cashflow"]["message"] = f"Loaded {len(cashflow_details)} cashflow records"
            else:
                status["cashflow"]["status"] = "error"
                status["cashflow"]["message"] = validation_message
                cashflow_details = None
        else:
            status["cashflow"]["status"] = "error"
            status["cashflow"]["message"] = f"Error reading cashflow file: {message}"
    
    # Load company data
    if company_url:
        status["company"]["status"] = "in_progress"
        company_insights, message = load_csv_from_drive_url(company_url)
        if company_insights is not None:
            is_valid, validation_message = validate_csv_file(company_insights, ['company_name'])
            if is_valid:
                status["company"]["status"] = "success"
                status["company"]["message"] = f"Loaded {len(company_insights)} company records"
            else:
                status["company"]["status"] = "error"
                status["company"]["message"] = validation_message
                company_insights = None
        else:
            status["company"]["status"] = "error"
            status["company"]["message"] = f"Error reading company file: {message}"
    
    # Create vector store
    if bond_details is not None or company_insights is not None:
        vector_store = create_vector_store(bond_details, company_insights)
        st.session_state.vector_store = vector_store
    
    return bond_details, cashflow_details, company_insights, status

def make_bond_documents(bond_details):
    """Convert bond data to documents for vector store"""
    documents = []
    for _, row in bond_details.iterrows():
        text = (f"ISIN: {row['isin']} Company: {row['company_name']} "
                f"Coupon Rate: {row.get('coupon_rate', 'N/A')}% "
                f"Maturity Date: {row.get('maturity_date', 'N/A')} "
                f"Credit Rating: {row.get('credit_rating', 'N/A')} "
                f"Security Type: {row.get('security_type', 'N/A')} "
                f"Face Value: {row.get('face_value', 'N/A')} "
                f"Issue Size: {row.get('issue_size', 'N/A')} "
                f"Listing Exchange: {row.get('listing_exchange', 'N/A')} "
                f"Trustee: {row.get('trustee', 'N/A')}")
        metadata = {
            "isin": row['isin'],
            "company_name": row['company_name'],
            "coupon_rate": float(row.get('coupon_rate', 0)),
            "maturity_date": row.get('maturity_date', 'N/A'),
            "credit_rating": row.get('credit_rating', 'N/A'),
            "security_type": row.get('security_type', 'N/A'),
            "face_value": row.get('face_value', 'N/A'),
            "issue_size": row.get('issue_size', 'N/A'),
            "listing_exchange": row.get('listing_exchange', 'N/A'),
            "trustee": row.get('trustee', 'N/A'),
            "type": "bond"
        }
        documents.append(Document(page_content=text, metadata=metadata))
    return documents

def make_company_documents(company_insights):
    """Convert company data to documents for vector store"""
    documents = []
    for _, row in company_insights.iterrows():
        text = (f"Company: {row['company_name']} "
                f"Sector: {row.get('sector', 'N/A')} "
                f"EPS: {row.get('eps', 'N/A')} "
                f"Rating: {row.get('rating', 'N/A')} "
                f"Current Ratio: {row.get('current_ratio', 'N/A')}")
        metadata = {
            "company_name": row['company_name'],
            "sector": row.get('sector', 'N/A'),
            "eps": row.get('eps', 'N/A'),
            "rating": row.get('rating', 'N/A'),
            "current_ratio": row.get('current_ratio', 'N/A'),
            "type": "company"
        }
        documents.append(Document(page_content=text, metadata=metadata))
    return documents

def create_vector_store(bond_details, company_insights):
    """Create FAISS vector store from bond and company data"""
    bond_docs = make_bond_documents(bond_details) if bond_details is not None else []
    company_docs = make_company_documents(company_insights) if company_insights is not None else []
    all_docs = bond_docs + company_docs
    if not all_docs:
        return None
    return FAISS.from_documents(all_docs, embeddings)

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

def perform_web_search(query, num_results=3):
    """Search the web using DuckDuckGo"""
    try:
        search = DuckDuckGoSearchAPIWrapper()
        results = search.results(query, num_results)
        return results
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        return []

def scrape_webpage(url):
    """Scrape content from a webpage"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text(separator='\n').strip()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text[:5000]
    except Exception as e:
        logger.error(f"Error scraping webpage {url}: {str(e)}")
        return f"Error scraping webpage: {str(e)}"

def get_bond_details(bond_details, isin=None):
    """Fetch bond details by ISIN"""
    if bond_details is None:
        return {"error": "Bond data not loaded"}
    if isin and isin in bond_details['isin'].values:
        row = bond_details[bond_details['isin'] == isin].iloc[0].to_dict()
        return row
    return {"error": f"Bond with ISIN {isin} not found"}

def get_cashflow(cashflow_details, isin):
    """Fetch cashflow data by ISIN"""
    if cashflow_details is None:
        return [{"error": "Cashflow data not loaded"}]
    if isin:
        cf_data = cashflow_details[cashflow_details['isin'] == isin]
        if not cf_data.empty:
            return cf_data.to_dict('records')
        return [{"error": f"No cashflow data for ISIN {isin}"}]
    return [{"error": "No ISIN provided"}]

def process_isin_lookup(vector_store, isin, company_name=None):
    """Handle ISIN lookup with company validation"""
    results = vector_store.search(f"ISIN: {isin}", search_type="mmr", k=1)
    if not results:
        return {"error": f"No bond found for ISIN {isin}"}
    
    bond = results[0].metadata
    if company_name and company_name.lower() not in bond["company_name"].lower():
        company_bonds = vector_store.search(f"Company: {company_name}", search_type="mmr", k=10, filter={"type": "bond"})
        return {
            "error": f"ISIN {isin} belongs to {bond['company_name']}, not {company_name}",
            "company_bonds": [{"isin": doc.metadata["isin"], "company_name": doc.metadata["company_name"]} for doc in company_bonds]
        }
    
    return {
        "isin": bond["isin"],
        "company_name": bond["company_name"],
        "coupon_rate": bond["coupon_rate"],
        "maturity_date": bond["maturity_date"],
        "credit_rating": bond["credit_rating"],
        "security_type": bond["security_type"],
        "face_value": bond["face_value"],
        "issue_size": bond["issue_size"],
        "listing_exchange": bond["listing_exchange"],
        "trustee": bond["trustee"]
    }

def process_company_issuances(vector_store, company_name):
    """List all bonds issued by a company"""
    results = vector_store.search(f"Company: {company_name}", search_type="mmr", k=50, filter={"type": "bond"})
    if not results:
        return {"error": f"No bonds found for {company_name}"}
    
    bonds = [{"isin": doc.metadata["isin"], "coupon_rate": doc.metadata["coupon_rate"], 
              "maturity_date": doc.metadata["maturity_date"], "face_value": doc.metadata["face_value"], 
              "credit_rating": doc.metadata["credit_rating"], "issue_size": doc.metadata["issue_size"]} 
             for doc in results]
    active = sum(1 for bond in bonds if pd.to_datetime(bond["maturity_date"], errors='coerce') > pd.Timestamp.now())
    return {"bonds": bonds, "total": len(bonds), "active": active, "matured": len(bonds) - active}

def process_filter_search(vector_store, query):
    """Handle filter-based bond searches"""
    filters = {}
    if "coupon rate above 10%" in query.lower():
        filters["coupon_rate"] = lambda x: x > 10
    if "maturity after 2026" in query.lower():
        filters["maturity_date"] = lambda x: pd.to_datetime(x, errors='coerce') > pd.Timestamp("2026-12-31")
    if "secured" in query.lower():
        filters["security_type"] = lambda x: "secured" in str(x).lower()
    
    results = vector_store.similarity_search_with_score(query, k=50, filter={"type": "bond"})
    filtered_bonds = [doc.metadata for doc, score in results if all(f(doc.metadata[key]) for key, f in filters.items())]
    return {"bonds": filtered_bonds[:5], "total": len(filtered_bonds)}

def process_cashflow_query(cashflow_details, isin):
    """Fetch cashflow schedule"""
    cashflow_data = get_cashflow(cashflow_details, isin)
    if "error" in cashflow_data[0]:
        return cashflow_data
    return {"cashflow": cashflow_data}

def process_screener_query(vector_store, company_name, metric=None):
    """Handle bond screener queries"""
    results = vector_store.search(f"Company: {company_name}", search_type="mmr", k=1, filter={"type": "company"})
    if not results:
        return {"error": f"No data found for {company_name}"}
    
    company = results[0].metadata
    if metric == "eps":
        return {"eps": company["eps"]}
    elif metric == "rating":
        return {"rating": company["rating"]}
    elif metric == "sector":
        return {"sector": company["sector"]}
    return {"company": company}

def process_bond_finder_query(query):
    """Handle bond finder queries with mock data"""
    bond_finder_data = [
        {"issuer": "Tata Capital", "rating": "AAA", "yield_range": "7.5%-8.0%", "platform": "SMEST"},
        {"issuer": "Indiabulls Housing Finance", "rating": "AA", "yield_range": "9.2%-9.8%", "platform": "FixedIncome & SMEST"}
    ]
    if "yield of more than 9%" in query.lower():
        return {"bonds": [b for b in bond_finder_data if float(b["yield_range"].split("%")[0].split("-")[-1]) > 9]}
    return {"bonds": bond_finder_data}

def process_calculator_query(query, bond_details, cashflow_details, isin, price=None):
    """Handle yield or price calculations (placeholder)"""
    bond = get_bond_details(bond_details, isin)
    if "error" in bond:
        return bond
    
    if price:
        coupon_rate = float(bond.get('coupon_rate', 0))
        simple_yield = (coupon_rate / price) * 100
        return {"yield": round(simple_yield, 2), "price": price}
    
    return {"error": "Please provide price for yield calculation"}

def process_query(query, bond_details, cashflow_details, company_insights, vector_store):
    """Determine query type and extract key information"""
    query_lower = query.lower()
    context = {"query": query, "query_type": "unknown"}
    
    isin = None
    company_name = None
    price = None
    for word in query.split():
        clean_word = ''.join(c for c in word if c.isalnum() or c in ".")
        if clean_word.upper().startswith("INE") and len(clean_word) >= 10:
            isin = clean_word.upper()
        if word.startswith("$"):
            try:
                price = float(word[1:])
            except ValueError:
                pass
    if "by" in query_lower:
        company_name = query_lower.split("by")[-1].strip()
    
    if isin:
        if "cash flow" in query_lower:
            context["query_type"] = "cashflow"
        elif "face value" in query_lower:
            context["query_type"] = "face_value"
        elif "yield" in query_lower or "calculate" in query_lower:
            context["query_type"] = "calculator"
        else:
            context["query_type"] = "isin_lookup"
    elif "issuances" in query_lower and company_name:
        context["query_type"] = "company_issuances"
    elif "debentures" in query_lower or "coupon rate" in query_lower or "maturity" in query_lower:
        context["query_type"] = "filter_search"
    elif "bond screener" in query_lower or "eps" in query_lower or "rating" in query_lower or "sector" in query_lower:
        context["query_type"] = "screener"
    elif "bond finder" in query_lower or "available" in query_lower or "yield" in query_lower:
        context["query_type"] = "bond_finder"
    
    context["isin"] = isin
    context["company_name"] = company_name
    context["price"] = price
    return context

def create_rag_chain(vector_store, llm):
    """Create RAG chain for query processing"""
    prompt_template = """You are a financial assistant specializing in bonds. Use the provided context to answer the query in a professional, friendly manner with Markdown formatting. Structure responses as tables, lists, or text as specified. If the context is insufficient, provide an error message politely.

    User Query: {question}
    Retrieved Context: {context}
    
    Format the response according to the query type (e.g., ISIN lookup, filter-based search, financial metric fetch). Include tables for lists and error handling for mismatched inputs.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return rag_chain

def generate_response(context, llm, rag_chain):
    """Generate formatted response based on query type"""
    if llm is None:
        return "Please enter a valid GROQ API key."
    if "error" in context:
        return f"**Error**: {context['error']}"
    
    query_type = context["query_type"]
    response = ""
    
    if query_type == "isin_lookup":
        bond = context.get("bond", {})
        if "error" in bond:
            response = f"**Error**: {bond['error']}"
            if "company_bonds" in bond:
                response += f"\n\n**{context['company_name']} Bonds**:\n| ISIN | Company |\n|------|---------|\n"
                for b in bond["company_bonds"]:
                    response += f"| {b['isin']} | {b['company_name']} |\n"
        else:
            response = f"""## Bond Details
- **ISIN**: {bond['isin']}
- **Issuer**: {bond['company_name']}
- **Coupon Rate**: {bond['coupon_rate']}%
- **Maturity Date**: {bond['maturity_date']}
- **Credit Rating**: {bond['credit_rating']}
- **Security Type**: {bond['security_type']}
- **Face Value**: {bond['face_value']}
- **Issue Size**: {bond['issue_size']}
- **Listing Exchange**: {bond['listing_exchange']}
- **Trustee**: {bond['trustee']}
"""
    elif query_type == "company_issuances":
        data = context.get("data", {})
        if "error" in data:
            response = f"**Error**: {data['error']}"
        else:
            response = f"""## {context['company_name']} Bond Issuances
- **Total Bonds**: {data['total']}
- **Active Bonds**: {data['active']}
- **Matured Bonds**: {data['matured']}
| ISIN | Coupon Rate | Maturity Date | Face Value | Credit Rating | Issue Size |
|------|-------------|---------------|------------|---------------|------------|
"""
            for bond in data["bonds"]:
                response += f"| {bond['isin']} | {bond['coupon_rate']}% | {bond['maturity_date']} | {bond['face_value']} | {bond['credit_rating']} | {bond['issue_size']} |\n"
    elif query_type == "filter_search":
        data = context.get("data", {})
        if "error" in data:
            response = f"**Error**: {data['error']}"
        else:
            response = f"""## Filtered Bonds
Found {data['total']} bonds matching criteria. Showing top 5:
| ISIN | Issuer | Coupon Rate | Maturity Date | Security Type |
|------|--------|-------------|---------------|---------------|
"""
            for bond in data["bonds"]:
                response += f"| {bond['isin']} | {bond['company_name']} | {bond['coupon_rate']}% | {bond['maturity_date']} | {bond['security_type']} |\n"
    elif query_type == "cashflow":
        cashflows = context.get("cashflow", [])
        if "error" in cashflows[0]:
            response = f"**Error**: {cashflows[0]['error']}"
        else:
            response = f"""## Cash Flow Schedule
| Date | Type |
|------|------|
"""
            for cf in cashflows:
                response += f"| {cf['cash_flow_date']} | {cf.get('cash_flow_type', 'Interest Payment')} |\n"
    elif query_type == "screener":
        data = context.get("data", {})
        if "error" in data:
            response = f"**Error**: {data['error']}"
        else:
            if "eps" in data:
                response = f"## {context['company_name']} Financials\n- **EPS**: {data['eps']}"
            elif "rating" in data:
                response = f"## {context['company_name']} Financials\n- **Rating**: {data['rating']}"
            elif "sector" in data:
                response = f"## {context['company_name']} Financials\n- **Sector**: {data['sector']}"
    elif query_type == "bond_finder":
        bonds = context.get("bonds", [])
        response = f"""## Bond Finder Results
| Issuer | Rating | Yield Range | Available at |
|--------|--------|-------------|--------------|
"""
        for bond in bonds:
            response += f"| {bond['issuer']} | {bond['rating']} | {bond['yield_range']} | {bond['platform']} |\n"
    elif query_type == "calculator":
        result = context.get("result", {})
        if "error" in result:
            response = f"**Error**: {result['error']}"
        else:
            response = f"## Yield Calculation\n- **Price**: ${result['price']}\n- **Yield**: {result['yield']}%"
    
    if not response:
        result = rag_chain.invoke({"query": context["query"]})
        response = result["result"]
        response += "\n\n**Sources**:\n" + "\n".join([f"- {doc.metadata.get('isin', doc.metadata.get('company_name', 'Web'))}" for doc in result["source_documents"]])
    
    return response

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
                st.session_state.bond_details, st.session_state.cashflow_details, st.session_state.company_insights, st.session_state.data_loading_status = load_data_from_drive(
                    bond_urls, cashflow_url, company_url
                )
                if (st.session_state.bond_details is not None or 
                    st.session_state.cashflow_details is not None or 
                    st.session_state.company_insights is not None):
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

    # Data status section
    st.markdown("### Data Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        bond_status = st.session_state.data_loading_status.get("bond", {"status": "not_started", "message": "Not loaded"})
        st.markdown(f"{display_status_indicator(bond_status['status'])} **Bond Data:** {bond_status['message']}")
    with col2:
        cashflow_status = st.session_state.data_loading_status.get("cashflow", {"status": "not_started", "message": "Not loaded"})
        st.markdown(f"{display_status_indicator(cashflow_status['status'])} **Cashflow Data:** {cashflow_status['message']}")
    with col3:
        company_status = st.session_state.data_loading_status.get("company", {"status": "not_started", "message": "Not loaded"})
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
            st.session_state.rag_chain = create_rag_chain(st.session_state.vector_store, llm)
        
        with st.spinner("Processing your query..."):
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            context = process_query(
                query, 
                st.session_state.bond_details,
                st.session_state.cashflow_details,
                st.session_state.company_insights,
                st.session_state.vector_store
            )
            
            query_type = context["query_type"]
            if query_type == "isin_lookup":
                context["bond"] = process_isin_lookup(
                    st.session_state.vector_store, context["isin"], context["company_name"])
            elif query_type == "company_issuances":
                context["data"] = process_company_issuances(
                    st.session_state.vector_store, context["company_name"])
            elif query_type == "filter_search":
                context["data"] = process_filter_search(st.session_state.vector_store, query)
            elif query_type == "cashflow":
                context["cashflow"] = process_cashflow_query(
                    st.session_state.cashflow_details, context["isin"])
            elif query_type == "screener":
                metric = "eps" if "eps" in query.lower() else "rating" if "rating" in query.lower() else "sector"
                context["data"] = process_screener_query(
                    st.session_state.vector_store, context["company_name"], metric)
            elif query_type == "bond_finder":
                context["bonds"] = process_bond_finder_query(query)
            elif query_type == "calculator":
                context["result"] = process_calculator_query(
                    query, st.session_state.bond_details, st.session_state.cashflow_details, 
                    context["isin"], context["price"])
            
            response = generate_response(context, llm, st.session_state.rag_chain)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

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
        st.write(f"- Bond Details: {display_status_indicator(st.session_state.data_loading_status['bond']['status'])} {st.session_state.data_loading_status['bond']['message']}")
        st.write(f"- Cashflow Details: {display_status_indicator(st.session_state.data_loading_status['cashflow']['status'])} {st.session_state.data_loading_status['cashflow']['message']}")
        st.write(f"- Company Insights: {display_status_indicator(st.session_state.data_loading_status['company']['status'])} {st.session_state.data_loading_status['company']['message']}")
        
        if st.checkbox("Show Data Samples"):
            if st.session_state.bond_details is not None:
                st.subheader("Bond Details Sample")
                st.dataframe(st.session_state.bond_details.head(3))
            if st.session_state.cashflow_details is not None:
                st.subheader("Cashflow Details Sample")
                st.dataframe(st.session_state.cashflow_details.head(3))
            if st.session_state.company_insights is not None:
                st.subheader("Company Insights Sample")
                st.dataframe(st.session_state.company_insights.head(3))

if __name__ == "__main__":
    main()
