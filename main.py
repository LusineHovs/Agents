from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.agents.factory import create_agent
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


load_dotenv()

tools = [TavilySearch(
    include_domains=[],
    exclude_domains=[]
)]
llm = ChatOllama(model="llama3.2")
react_prompt = PromptTemplate.from_template("""
You are a helpful assistant that can use tools to answer questions.

You have access to the following tools:
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

Question: {input}
Thought: {agent_scratchpad}
""")

# creating reasoning engine using modern agent factory
agent = create_agent(model=llm, tools=tools, system_prompt=react_prompt.template)
chain = agent

def main():
    result = chain.invoke({"messages": [{"role": "user", "content": "Search for job positions for an ai engineer using langchain in the bay area on linkedin and list their details"}]})
    print(result)


if __name__ == "__main__":
    main()
