import os
import warnings
import requests
import certifi
import streamlit as st
from dotenv import load_dotenv

# Suppress deprecation warnings
warnings.filterwarnings("ignore")

from langchain_groq import ChatGroq
from langchain.tools import tool

try:
    from langchain.agents import create_react_agent, AgentExecutor
except ImportError:
    from langchain_classic.agents import create_react_agent, AgentExecutor

try:
    from langchain import hub
except ImportError:
    from langchain_classic import hub

try:
    from langchain_tavily import TavilySearch as TavilySearchResults
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults

# ==========================================
# ENV & SSL SETUP
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ==========================================
# PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling for Clean Modern UI
st.markdown("""
    <style>
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Header card styling */
    .header-card {
        background: rgba(30, 41, 59, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 15px;
    }
    
    /* Tool Badges */
    .tool-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }

    /* Response box styling */
    .response-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-left: 5px solid #38bdf8;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        color: #f8fafc;
        font-size: 1.05rem;
        line-height: 1.6;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# AGENT & TOOLS DEFINITION
# ==========================================

@tool
def get_weather_data(city: str) -> str:
    """Fetch current weather information for a given city."""
    if not WEATHERSTACK_API_KEY:
        return "Weatherstack API key is missing. Please set WEATHERSTACK_API_KEY in .env file."

    url = f"http://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={city}"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "current" not in data:
            error_info = data.get("error", {}).get("info", "Could not fetch weather data")
            return f"Could not fetch weather data for {city}: {error_info}"

        current = data["current"]
        location = data.get("location", {})

        return (
            f"City: {location.get('name', city)}, {location.get('country', '')}\n"
            f"Temperature: {current.get('temperature')}°C\n"
            f"Weather Description: {', '.join(current.get('weather_descriptions', []))}\n"
            f"Humidity: {current.get('humidity')}%\n"
            f"Wind Speed: {current.get('wind_speed')} km/h"
        )
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

search_tool = TavilySearchResults(max_results=3)
tools = [search_tool, get_weather_data]

# Initialize Pure Groq Llama 3.3 Model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY
)

# Pull prompt from hub with fallback template
try:
    prompt = hub.pull("hwchase17/react")
except Exception:
    from langchain_core.prompts import PromptTemplate
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    prompt = PromptTemplate.from_template(template)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# ==========================================
# MAIN INTERFACE
# ==========================================

st.markdown("""
    <div class="header-card">
        <div class="header-title">⚡ Agentic AI Assistant</div>
        <div class="header-subtitle">Autonomous ReAct Agent powered by Groq Llama 3.3 70B</div>
        <div>
            <span class="tool-badge">⚡ Groq Llama 3.3</span>
            <span class="tool-badge">🌐 Tavily Search</span>
            <span class="tool-badge">🌤️ Weather API</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sample Queries Section
st.subheader("💡 Quick Examples")
col1, col2, col3 = st.columns(3)

example_prompt = ""
if col1.button("📌 Weather in Tokyo", use_container_width=True):
    example_prompt = "What is the current weather in Tokyo?"
if col2.button("📌 India Capital & Weather", use_container_width=True):
    example_prompt = "Find the capital of India and its current weather."
if col3.button("📌 Latest AI News", use_container_width=True):
    example_prompt = "Search for the latest news in Artificial Intelligence."

# User Input Form
user_query = st.text_input(
    "Ask the Agent anything:",
    value=example_prompt if example_prompt else "",
    placeholder="e.g. Find the capital of France and get its current weather"
)

run_button = st.button("🚀 Run Agent", type="primary", use_container_width=True)

if run_button:
    if not user_query.strip():
        st.warning("⚠️ Please enter a query before running the agent.")
    elif not GROQ_API_KEY:
        st.error("🔑 GROQ_API_KEY missing! Please set GROQ_API_KEY in your .env file.")
    else:
        with st.spinner("🤖 Agent is thinking & searching..."):
            try:
                response = agent_executor.invoke({"input": user_query})
                output_text = response.get("output", "No response generated.")

                st.success("✅ Complete!")
                st.markdown("### 📝 Response")
                st.markdown(f'<div class="response-card">{output_text}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                err_str = str(e)
                st.error(f"❌ Error running agent: {err_str}")