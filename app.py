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

# --- 3. Gemini LLM call ---
def call_gemini_llm(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"❌ Gemini API call failed: {e}")
        if 'API key' in str(e):
            st.error("Please ensure your Gemini API key is correct and valid in your Streamlit secrets.")
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
        if df[col].dtype == 'object':
            examples = df[col].unique()[:3]
            sample = ", ".join([f"'{str(x)}'" for x in examples])
            schema_parts.append(f"{col} (object, e.g., {sample})")
        else:
            schema_parts.append(f"{col} ({df[col].dtype})")
    return "\n".join(schema_parts)

def generate_llm_prompt(query, df):
    schema = get_data_schema(df)
    return f"""
You are an expert Python data analyst assistant. Your task is to generate Python code to answer a user's question about a pandas DataFrame named `df`.

**DATAFRAME SCHEMA:**
"{schema}"
**USER QUERY:**
"{query}"

**INSTRUCTIONS:**
1. Use `df` directly. Do not reload or open files.
2. Assign the result (scalar, DataFrame, or Series) to a variable named `result`.
3. For plots, use `matplotlib.pyplot`, and display with `st.pyplot(plt.gcf())`. Do not call `plt.show()`.
4. Normalize object columns for comparisons using `.str.lower().str.strip()`.
5. Only return executable Python code in a code block.
"""

# --- 5. Code Execution ---
def execute_generated_code(code, df):
    df_copy = df.copy()
    local_scope = {'df': df_copy, 'pd': pd, 'plt': plt, 'st': st}
    try:
        exec(code, {}, local_scope)
        result = local_scope.get("result", None)
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return result
        elif isinstance(result, (int, float, bool, str)):
            return str(result)
        return "✅ Code executed successfully. If you asked for a chart, it should be displayed above."
    except Exception as e:
        st.error("❌ An error occurred while executing the generated code.")
        st.exception(e)
        return None

# --- 6. Streamlit App UI ---
st.set_page_config(page_title="NeoAT Excel Assistant", layout="centered")
st.title("Tanmay's Excel Sheet Analyzer")
st.markdown("Ask questions like:\n- *‘Count countries with lower rank than Syria’*\n- *‘Plot bar chart of top 5 by score’*")

with st.sidebar:
    st.header("Controls")
    api_key = st.text_input("🔑 Gemini API Key", type="password")
    uploaded_file = st.file_uploader("📁 Upload your Excel file", type=["xlsx"])
    query = st.text_input("❓ Ask a question about your data")
    submit_button = st.button("🚀 Analyze")

if submit_button:
    if not api_key:
        st.warning("🔑 Please enter your Gemini API key.")
    elif not uploaded_file:
        st.warning("📁 Please upload an Excel file.")
    elif not query:
        st.warning("❓ Please enter a question.")
    else:
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            df = normalize_column_names(df)
            df = normalize_string_values(df)

            st.subheader("🔍 Data Preview")
            st.dataframe(df.head())

            prompt = generate_llm_prompt(query, df)

            with st.spinner("🤖 Thinking with Gemini..."):
                response = call_gemini_llm(prompt, api_key)
                if response:
                    code = extract_python_code(response)

                    st.subheader("🧾 Generated Python Code")
                    st.code(code, language="python")

                    st.subheader("💡 Answer / Chart")
                    result = execute_generated_code(code, df.copy())

                    if isinstance(result, (pd.DataFrame, pd.Series)):
                        st.dataframe(result)
                    elif result is not None:
                        st.write(result)

        except Exception as e:
            st.error(f"❌ Failed to process: {e}")
