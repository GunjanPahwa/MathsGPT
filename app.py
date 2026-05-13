from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool                                
from langchain_core.prompts import PromptTemplate
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent

st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant")
st.title("Text to Math Problem Solver using LLama 3")

groq_api_key = st.sidebar.text_input(label="GROQ API KEY", type="password")

if not groq_api_key:
    st.info("Please add your groq api key to continue")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

wikipedia_wrapper = WikipediaAPIWrapper()

prompt = """
You are an agent tasked for solving users mathematical questions. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""
prompt_template = PromptTemplate(input_variables=["question"], template=prompt)
chain = prompt_template | llm | StrOutputParser()

# ✅ Tools defined with @tool decorator
@tool
def wikipedia_search(query: str) -> str:
    """A tool for searching information on any topic."""
    return wikipedia_wrapper.run(query)

@tool
def calculator(expression: str) -> str:
    """A tool for answering math related questions. Only input mathematical expressions as python code."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def reasoning_tool(question: str) -> str:
    """A tool for answering logic based and reasoning questions."""
    return chain.invoke({"question": question})

assistant_agent = create_react_agent(llm, tools=[wikipedia_search, calculator, reasoning_tool])

def convert_messages(messages):
    converted = []
    for msg in messages:
        if msg["role"] == "user":
            converted.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            converted.append(AIMessage(content=msg["content"]))
    return converted

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.text_area("Enter your question:", "")
if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            response = assistant_agent.invoke(
                {"messages": convert_messages(st.session_state.messages)}
            )
            response = response["messages"][-1].content

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("Response:")
            st.success(response)
    else:
        st.warning("Please enter the input")
