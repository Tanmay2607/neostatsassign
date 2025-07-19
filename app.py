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
1. Generate only executable Python code that uses the 'df' DataFrame.
2. Do NOT include explanations or text outside the code block.
3. 3. The code must calculate the answer and assign it to a variable named `result`. 
   - If the query involves listing rows, assign the filtered DataFrame to `result`.
   - If it involves a value (like sum or count), assign the scalar to `result`.
4. If the query requires a visualization (e.g., "bar chart", "histogram"), generate valid code to create the plot using matplotlib.
   Use `st.pyplot(plt.gcf())` instead of `plt.show()` to display the chart in Streamlit.
5. Assume 'df' is already loaded.
6. When filtering by a text column, always use `.str.lower().replace(",", "").str.strip()` — never use `.lower()` directly on a Series or DataFrame.
7. Always check `.empty` before accessing `.iloc[0]` to avoid index errors.
8. Never use `.empty` on a string or scalar. Use `.empty` only on DataFrames or Series.
9. Never load data from a file. Do not use `pd.read_csv`, `pd.read_excel`, or any other file operations. The DataFrame named `df` is already loaded. Use it directly.
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
