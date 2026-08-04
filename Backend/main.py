import os
import warnings
import requests
import certifi
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
# LOAD ENV VARIABLES
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ==========================================
# WEATHER TOOL
# ==========================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """
    if not WEATHERSTACK_API_KEY:
        return "Weatherstack API key is missing. Please set WEATHERSTACK_API_KEY in .env file."

    url = (
        f"http://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

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

# ==========================================
# TOOLS & LLM SETUP
# ==========================================
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool, get_weather_data]

# Pure Groq LLM Setup
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
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# ==========================================
# RUN CLI AGENT
# ==========================================
if __name__ == "__main__":
    query = "Find the capital of India and then find its current weather."
    print(f"Running agent with query: {query}\n")
    try:
        response = agent_executor.invoke({"input": query})
        print("\n========================")
        print("FINAL OUTPUT:")
        print("========================")
        print(response.get("output"))
    except Exception as e:
        print(f"\nError running agent: {e}")