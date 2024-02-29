
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, create_openai_tools_agent
from langchain.agents import  Tool
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
import streamlit as st
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_experimental.tools import PythonREPLTool
#import matplotlib.pyplot as plt
import os
from langchain_community.utilities import SerpAPIWrapper
from csv_sfdc import update_survey_to_salesforce_account

st.set_page_config(
    page_title="ServiceNow Knowledge Base Chatbot",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# Customer survey chatbot🔗"


#added for azure openai
# Configure the baseline configuration of the OpenAI library for Azure OpenAI Service.
#os.environ["OPENAI_API_TYPE"] = "azure"
#os.environ["OPENAI_API_VERSION"] = "2023-05-15"
#os.environ["AZURE_OPENAI_ENDPOINT"] = "https://azopenai-123.openai.azure.com/"
#os.environ["OPENAI_API_KEY"] = constants.key
#os.environ["SERPAPI_API_KEY"] = constants.SERP_API_KEY


search = SerpAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions from the web, related to service contracts survey. You should ask targeted questions.Use it only after you have tried using the PythonREPLTool.",
    ),
    PythonREPLTool(),
    ]

tools.append(update_survey_to_salesforce_account)

llm = AzureChatOpenAI(
    deployment_name="gpt4-test",
    model_name="gpt-4",
)

#Set up System Prompt template
message= SystemMessage(
    content= (
    """
        You are a chatbot having a conversation with a human regarding service contracts.
        You have the following tools to use:
            PythonREPLTool: First you rephrase the user input so that a Python code generator can understand it.
            You have access to python REPL, which you can use to execute python code.
            Generate python code to answer the user input based on service_contracts_survey_data.csv file which has the columns: 
            
                ID
                Name
                Contact Number
                Email
                Year
                Make
                Model
                City
                State
                Zip Code
                Feedback
                How did you hear about our vehicle extended service contracts?
                How easy was it to purchase your extended service contract?
                How satisfied are you with the coverage options provided?
                Rate the clarity of information provided regarding what is and isnt covered under your service contract.
                Have you had to use your extended service contract for vehicle repairs?
                How easy was it to file a claim under your extended service contract?
                How satisfied were you with the speed of claim processing?
                Rate the quality of repair service received.
                How would you rate the customer service you received?
                How likely are you to renew your extended service contract?
                How likely are you to recommend our extended service contracts to others?
                How would you rate your experience with our self-care web and mobile app in managing your extended service contract?
                What features or functionalities would you like to see improved or added to our self-care web and mobile app?
                What aspects of our service and contract options can be improved?
            
            If the user asks you to plot a graph, get the data you need for it, generate and save the image as graph.png.
            For any kind of sentiment analysis or analysis on the Feedback column, use the textblob python library.
            If you get an error, debug your code and try again.
            Only use the output of your code to answer the question.
            You might know the answer without running any code, but you should still run the code to get the answer.Use this to chat with the user regarding customer survey data.
        
        search tool: useful for when you need to answer questions from the web, related to service contracts survey. 
        You should ask targeted questions.Use it only after you have tried using the PythonREPLTool. When doing search, cite the source/link of the website from where it is getting the information
        Provide the links from the best searches on the web mandatorily.

        Update Salesforce tool: This tool will update the Survey Response Data to Salesforce Account Object
        
    """
    )
    )

#Create Prompt
prompt=  OpenAIFunctionsAgent.create_prompt (
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],

)

#setup Agent
agent=create_openai_tools_agent(llm=llm, prompt=prompt, tools=tools) 

#setup Agent executor
agent_executor= AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True


)

memory = ConversationBufferMemory(llm=llm,memory_key='history', return_messages=True, output_key='output')
#matplotlib.use('TkAgg')
#plt.close('all')
starter_message = "Ask me questions about the Survey Response!"
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]
    
    
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)
    

if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor(
            {"input": prompt, "history": st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        if os.path.exists('graph.png'):
            st.image("graph.png")
        memory.save_context({"input": prompt}, response)
        if os.path.exists('graph.png'):
            os.remove('graph.png')
        st.session_state["messages"] = memory.buffer
        run_id = response["__run"].run_id