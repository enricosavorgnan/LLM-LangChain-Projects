# ------------------------------------------------------------------ #
# This code uses libraries LangChain, WolframAlpha and OpenAI to     #
# make GPT communicating with Wolfran engine.                        #
# This could be useful to get scientific results and answers to      #
# mathematical problems.                                             #
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
# You require the following libraries:
# · langchain
# · openai
# · wolframalpha
# ------------------------------------------------------------------ #



from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
import os

# ------------------------------------------------------------------ #
# Setting the environment
# ------------------------------------------------------------------ #
os.environ["OPENAI_API_KEY"] = "    your_openai_api_key_goes_here      "
os.environ["WOLFRAM_ALPHA_APPID"] = '   your_wolfram_alpha_appid_goes_here     '


# ------------------------------------------------------------------ #
# Let's play
# ------------------------------------------------------------------ #
llm = OpenAI(temperature=0)
tools = load_tools(['wolfram-alpha'])
agent = initialize_agent(tools, llm, verbose=True, agent='zero-shot-react-description')
agent.run((input()))