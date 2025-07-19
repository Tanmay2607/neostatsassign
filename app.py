# --- 1. Imports ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import re

# --- 2. Helper: Extract code from LLM ---
def extract_python_code(response_text):
    """Extracts Python code from a Markdown code block."""
    # Updated regex to be more robust
    match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback for responses that might not use the 'python' tag
    match = re.search(r"```(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()

# --- 3. Gemini LLM call ---
def call_gemini_llm(prompt, api_key):
    """Sends a prompt to the Gemini API and returns the text response."""
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

# --- 4. Schema, Normalization, and Prompt Generation ---
def normalize_column_names(df):
    """Cleans and standardizes all column names in the DataFrame."""
    new_cols = {}
    for col in df.columns:
        # Replace special characters and spaces with underscore, convert to lowercase
        clean_col = re.sub(r'[^0-9a-zA-Z]+', '_', col).lower()
        # Remove leading/trailing underscores
        clean_col = clean_col.strip('_')
        new_cols[col] = clean_col
    df = df.rename(columns=new_cols)
    return df

def get_data_schema(df):
    """Generates a string representation of the DataFrame's schema."""
    schema_parts = []
    for col in df.columns:
        # Providing more context for object columns
        if df[col].dtype == 'object':
            unique_examples = df[col].unique()[:3] # Show up to 3 unique examples
            example_str = ", ".join([f"'{str(x)}'" for x in unique_examples])
            schema_parts.append(f"{col} (object, e.g., {example_str})")
        else:
            schema_parts.append(f"{col} ({df[col].dtype})")
    return "\n".join(schema_parts)

def generate_llm_prompt(query, df):
    """Generates a detailed, schema-agnostic prompt for the LLM."""
    schema = get_data_schema(df)
    return f"""
You are an expert Python data analyst assistant. Your task is to generate Python code to answer a user's question about a pandas DataFrame named `df`.

**CONTEXT:**
- The `df` DataFrame is already loaded in memory.
- You must not load any data or use file operations.
- The user's query is a natural language question.
- You do not know the schema of `df` in advance. Rely ONLY on the schema provided below.

**DATAFRAME SCHEMA:**
"{schema}"
**USER QUERY:**
"{query}"

**INSTRUCTIONS:**
1.  **Analyze the Query and Schema:** Carefully consider the user's query and the provided data schema to understand the intent.
2.  **Generate Python Code:** Write a Python script that uses the `df` DataFrame to answer the query.
3.  **Output Assignment:**
    - For queries that result in a number, string, or boolean (e.g., "average age", "count of rows"), assign the final scalar value to a variable named `result`.
    - For queries that result in a list of items (e.g., "show all rows where country is 'USA'"), assign the resulting DataFrame or Series to the `result` variable.
4.  **Visualizations:**
    - If the query asks for a chart (e.g., "plot a bar chart", "show the distribution"), generate a plot using `matplotlib.pyplot`.
    - **DO NOT** use `plt.show()`. Instead, use `st.pyplot(plt.gcf())` to display the plot.
    - Ensure all plots are high quality: include a clear title, and label the x and y axes.
5.  **Code Quality and Safety:**
    - Write clean, efficient, and correct pandas code.
    - **Never hardcode column names.** Your code must be flexible and work even if the column names were different. Use the names provided in the schema above.
    - Assume columns of type 'object' might contain mixed case or leading/trailing spaces. When filtering on these columns, always normalize the comparison value and the column, like this: `df[df['column_name'].str.lower().str.strip() == 'some value']`.
    - Handle potential missing values (`NaN`) gracefully. Standard aggregations like `.mean()`, `.sum()`, and `.count()` do this automatically.

**Your response MUST be ONLY the executable Python code inside a markdown block.**
Do not include any explanations, comments outside the code, or introductory text.
"""

def execute_generated_code(code, df):
    """Executes the generated Python code in a controlled environment."""
    # Use a copy of the dataframe to prevent modification by the executed code
    df_copy = df.copy()
    local_scope = {'df': df_copy, 'pd': pd, 'plt': plt, 'st': st}

    try:
        exec(code, {}, local_scope)
        result = local_scope.get("result", None)

        # Handle various types of results
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return result
        elif isinstance(result, (int, float, bool, str)):
            return str(result)
        # If no 'result' variable is assigned, it implies a plot was generated.
        return "✅ Code executed successfully. If you asked for a chart, it should be displayed above."

    except Exception as e:
        st.error("❌ An error occurred while executing the generated code.")
        st.exception(e)
        return None

# --- 5. Streamlit App ---
st.set_page_config(page_title="Data Analysis Assistant", layout="wide")
st.title("💡 General-Purpose Data Analysis Assistant")
st.markdown("Upload any structured Excel file and ask questions in plain English. The AI will generate and run the code to answer you.")

# Use a sidebar for controls
with st.sidebar:
    st.header("Controls")
    api_key = st.text_input("Enter your Gemini API Key", type="password")
    uploaded_file = st.file_uploader("📁 Upload your Excel file", type=["xlsx"])
    query = st.text_input("❓ Ask a question about your data")
    submit_button = st.button("🚀 Analyze")
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    query = st.text_input("Ask a question about your excel sheet")
    submit_button = st.button("Analyzing")

if submit_button and not api_key:
    st.warning("🔑 Please enter your Gemini API key in the sidebar.")
elif submit_button and not uploaded_file:
    st.warning("📁 Please upload an Excel file.")
    st.warning("Please upload an Excel file.")
elif submit_button and not query:
    st.warning("❓ Please ask a question.")
    st.warning("Please ask a question.")
elif submit_button and api_key and uploaded_file and query:
    try:
        df_original = pd.read_excel(uploaded_file, engine="openpyxl")
        df_processed = normalize_column_names(df_original)

        st.subheader("🔍 Data Preview (Normalized)")
        st.subheader("Data Preview (Normalized)")
        st.dataframe(df_processed.head())

        prompt = generate_llm_prompt(query, df_processed)

        with st.spinner("🤖 AI is analyzing your data and writing code..."):
        with st.spinner("AI is analyzing your data and writing code..."):
            response_text = call_gemini_llm(prompt, api_key)

        if response_text:
            code_to_execute = extract_python_code(response_text)
            
            with st.expander("🧾 View Generated Python Code", expanded=False):
                st.code(code_to_execute, language="python")
        

            st.subheader("💡 Answer / Chart")
            st.subheader("Answer / Chart")
            result = execute_generated_code(code_to_execute, df_processed)

            if result is not None:
