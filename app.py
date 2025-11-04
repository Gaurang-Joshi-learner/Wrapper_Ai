import streamlit as st
import asyncio
import json, os
from llm_wrappers import query_all_llms

# -----------------------------
#  PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Wrapper.ai - Multi-LLM Comparator",
    layout="wide",
    page_icon="🤖"
)

# -----------------------------
#  PROMPT HISTORY FUNCTIONS
# -----------------------------
HISTORY_FILE = "prompt_history.json"

def save_prompt(prompt):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    if prompt not in [h["prompt"] for h in history]:
        history.append({"prompt": prompt})
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-10:], f, indent=2)

def load_prompts():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return [h["prompt"] for h in json.load(f)]
    return []

# -----------------------------
#  APP HEADER
# -----------------------------
st.title("🤖 Wrapper.ai")

# -----------------------------
#  THEME TOGGLE
# -----------------------------
with st.sidebar:
    theme = st.radio("🎨 Choose Theme", ["Light", "Dark"], index=0)
    st.markdown(
        f"""
        <style>
        body {{
            background-color: {'#0E1117' if theme=='Dark' else 'white'};
            color: {'white' if theme=='Dark' else 'black'};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
#  PROMPT HISTORY IN SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("📜 Prompt History")
    history = load_prompts()
    selected_prompt = st.selectbox("Pick a previous prompt", [""] + history)
    if selected_prompt:
        prompt = selected_prompt
    else:
        prompt = ""

# -----------------------------
#  MAIN INPUT AREA
# -----------------------------
prompt = st.text_area(
    "📝 Enter your prompt:",
    value=prompt if prompt else "",
    placeholder="Explain quantum computing in simple terms...",
    height=120
)
temperature = st.slider("🎚 Temperature", 0.0, 1.0, 0.7)
max_tokens = st.slider("🔠 Max Tokens", 100, 2000, 500)

# -----------------------------
#  GENERATE RESPONSES
# -----------------------------
if st.button("Generate Responses", type="primary"):
    if not prompt.strip():
        st.warning("⚠ Please enter a prompt before generating responses.")
    else:
        save_prompt(prompt)
        with st.spinner("Querying all models... ⏳"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            responses = loop.run_until_complete(
                query_all_llms(prompt, temperature, max_tokens)
            )

        # -----------------------------
        # 📊 ANALYTICS SECTION
        # -----------------------------
        avg_time = sum(r["time_ms"] for r in responses if r["time_ms"] > 0) / len(responses)
        st.success(f"⚡ Average response time: {avg_time:.2f} ms")

        # -----------------------------
        # 🧩 DISPLAY MODEL RESPONSES
        # -----------------------------
        st.markdown("### 🧾 Model Responses")
        cols = st.columns(len(responses))

        for i, col in enumerate(cols):
            with col:
                model_name = responses[i]["model"]
                response_text = responses[i]["response"]

                st.subheader(model_name)
                st.caption(f"Tokens: {responses[i]['tokens']} | Time: {responses[i]['time_ms']} ms")

                # Display response in scrollable text area
                st.text_area("Response", value=response_text, height=250, key=f"resp_{i}")

                # Copy button (JS-based)
                safe_text = response_text.replace("`", "'").replace('"', "'")
                copy_button = f"""
                    <button onclick="navigator.clipboard.writeText(`{safe_text}`)" 
                            style="background-color:#4CAF50;color:white;border:none;
                                   padding:6px 10px;text-align:center;display:inline-block;
                                   font-size:13px;margin:4px 2px;cursor:pointer;
                                   border-radius:6px;">
                        📋 Copy
                    </button>
                """
                st.markdown(copy_button, unsafe_allow_html=True)

                # Regenerate specific model
                if st.button(f"🔄 Regenerate {model_name}", key=f"regen_{i}"):
                    with st.spinner(f"Regenerating {model_name}..."):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        new_response = loop.run_until_complete(
                            query_all_llms(prompt, temperature, max_tokens)
                        )[i]
                        st.text_area("New Response", value=new_response["response"], height=250, key=f"new_{i}")
                        st.caption(f"Tokens: {new_response['tokens']} | Time: {new_response['time_ms']} ms")

# -----------------------------
# 🧾 FOOTER
# -----------------------------
st.markdown("---")

