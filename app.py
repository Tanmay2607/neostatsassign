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
        df[col] = df[col].astype(str).str.strip().str.lower()
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
1. Use the provided DataFrame `df`. Do not read from or write to any files.
2. Normalize object/string columns with `.str.strip().str.lower()` before comparisons.
3. Assign your final result (DataFrame, Series, value, or plot) to a variable named `result`.
4. If the answer is a chart, use matplotlib and call `st.pyplot(plt.gcf())`. Do not call `plt.show()`.
5. Return only executable Python code inside a code block.
"""

# --- 5. Code Execution ---
def execute_generated_code(code, df):
    df_copy = df.copy()
    local_scope = {'df': df_copy, 'pd': pd, 'plt': plt, 'st': st}
    try:
        exec(code, {}, local_scope)
        return local_scope.get("result", "✅ Code executed successfully.")
    except Exception as e:
        st.error("❌ An error occurred while executing the generated code.")
        st.exception(e)
        return None

# --- 6. Streamlit UI ---
st.set_page_config(page_title="NeoAT Excel Assistant", layout="centered")
st.title("📊 Tanmay's Excel Sheet Analyzer")

# Auto-read API key from secrets
api_key = st.secrets.get("GEMINI_API_KEY", "")

uploaded_file = st.file_uploader("📁 Upload your Excel file", type=["xlsx"])
query = st.text_input("💬 Ask a question about your Excel data")
submit_button = st.button("🔍 Analyze")

if submit_button:
    if not api_key:
        st.warning("🔐 Gemini API key missing in `secrets.toml`. Please add `GEMINI_API_KEY`.")
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
