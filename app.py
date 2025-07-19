# --- 1. Imports ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import re

# --- 2. Helper: Extract code from LLM ---
def extract_python_code(response_text):
    match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()

# --- 3. Gemini LLM Call ---
def call_gemini_llm(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"❌ Gemini API call failed: {e}")
        return None

# --- 4. Schema & Prompt ---
def normalize_column_names(df):
    new_cols = {}
    for col in df.columns:
        clean_col = re.sub(r'[^0-9a-zA-Z]+', '_', col).lower().strip('_')
        new_cols[col] = clean_col
    return df.rename(columns=new_cols)

def normalize_string_values(df):
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype(str).str.strip().str.lower().str.replace(",", "", regex=False)
    return df

def get_data_schema(df):
    schema_parts = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == 'object':
            examples = df[col].dropna().unique()[:3]
            sample = ", ".join([f"'{str(x)}'" for x in examples])
            schema_parts.append(f"{col} (object, e.g., {sample})")
        else:
            schema_parts.append(f"{col} ({dtype})")
    return "\n".join(schema_parts)

def generate_llm_prompt(query, df):
    schema = get_data_schema(df)
    return f"""
You are a Python data analyst assistant. Write Python code to answer the user's question based on the DataFrame `df`.

**DATAFRAME SCHEMA:**
{schema}

**USER QUESTION:**
{query}

**RULES:**
1. Generate only Python code. Do NOT return explanations or markdown — only a code block.
2. Use the DataFrame `df` which is already loaded.
3. Clean text columns using `.str.lower().replace(",", "").str.strip()` before filtering.
4. Assign the answer to `result`.
   - If scalar (e.g., count, mean), assign scalar to `result`.
   - If filtered rows, assign DataFrame to `result`.
5. If a chart is required, use matplotlib and display it with `st.pyplot(plt.gcf())`. Do NOT use `plt.show()`.
6. Avoid loading any files. Do not use `read_csv`, `read_excel`, etc.
7. If filtering by value, ensure string matching is lowercase and stripped.
8. Check `.empty` before using `.iloc[0]` to avoid errors.
"""

# --- 5. Code Execution ---
def execute_generated_code(code, df):
    df_copy = df.copy()
    local_scope = {'df': df_copy, 'pd': pd, 'plt': plt, 'st': st}
    try:
        exec(code, {}, local_scope)
        return local_scope.get("result", "✅ Code executed successfully.")
    except Exception as e:
        st.error("❌ Error during code execution.")
        st.exception(e)
        return None

# --- 6. Streamlit UI ---
st.set_page_config(page_title="NeoAT Excel Assistant", layout="centered")
st.title("📊 Tanmay's Excel Sheet Analyzer")

api_key = st.secrets.get("GEMINI_API_KEY", "")

uploaded_file = st.file_uploader("📁 Upload your Excel file", type=["xlsx"])
query = st.text_input("💬 Ask a question about your Excel data")
submit_button = st.button("🔍 Analyze")

if submit_button:
    if not api_key:
        st.warning("🔐 Missing Gemini API key in `secrets.toml`.")
    elif not uploaded_file:
        st.warning("📄 Please upload a valid Excel file.")
    elif not query:
        st.warning("❓ Please enter a question.")
    else:
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            df = normalize_column_names(df)
            df = normalize_string_values(df)

            st.subheader("📑 Data Preview")
            st.dataframe(df.head())

            prompt = generate_llm_prompt(query, df)
            with st.spinner("🤖 Thinking with Gemini..."):
                response = call_gemini_llm(prompt, api_key)
                if response:
                    code = extract_python_code(response)
                    result = execute_generated_code(code, df)

                    st.subheader("💡 Answer / Chart")
                    if isinstance(result, (pd.DataFrame, pd.Series)):
                        st.dataframe(result)
                    elif result is not None:
                        st.write(result)

        except Exception as e:
            st.error(f"❌ Failed to analyze your file: {e}")
