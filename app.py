import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain_community.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
import os
import pickle
from serpapi import GoogleSearch
from dotenv import load_dotenv
from typing import List, Union

#Google Sheets API setup
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
load_dotenv()
#Function to get Google Sheets credentials
def get_gsheets_credentials():
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret_126784372966-iias2i7ec3o4jsg9pvuo4rh36j9vnakq.apps.googleusercontent.com.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return creds

#Function to read Google Sheets data
def read_gsheet(sheet_id, sheet_range):
    creds = get_gsheets_credentials()
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=sheet_id, range=sheet_range).execute()
    values = result.get("values", [])
    if not values:
        return None
    else:
        return pd.DataFrame(values[1:], columns=values[0])

def perform_web_search(entity, custom_prompt):
    search_query = custom_prompt.format(company= entity)
    params = {
        "q": search_query,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    #Error handling for API request
    if "error" in results:
        st.error(f"Error in search for {entity}: {results['error']}")
        return []
    
    #Extract organic results
    organic_results = results.get("organic", [])
    
    #format results (example : get title and link)
    formatted_results = []
    for result in organic_results:
        formatted_results.append({
            "title": result.get("title"),
            "link": result.get("link"),
            "snippet": result.get("snippet")
        })
    return formatted_results



#setup LLM (Gemini 1.5 Pro)
llm  = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=os.getenv("GEMINI_API_KEY"))

#setup SerpAPI Wrapper
serpapi_wrapper = SerpAPIWrapper()

#setup tools
tools = [
    Tool(
        name="Search",
        func=perform_web_search,  #my web serach function
        description="useful when you need to search for information on the web"
    )
]


#Custom prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f'{tool.name}: {tool.description}' for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
    
#Custom output parser
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: {llm_output}")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    

# Initialize agent
prompt_template = CustomPromptTemplate(
    template="""
    You are a helpful AI assistant tasked with extracting information from web search results.

    For the entity {entity}, you need to find: {custom_prompt}.

    Here are the web search results:
    {search_results}

    Using these results, extract the following: {backend_prompt}

    Do not hallucinate. Do not make up factual information. Preserve the main result input tone. 
    If you cannot find the answer, respond with 'Information not found.'

    Tools:
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
    {agent_scratchpad}
    """,
    tools=tools,
    input_variables=["input", "entity", "custom_prompt", "backend_prompt", "search_results", "agent_scratchpad"]
)

output_parser = CustomOutputParser()

llm_chain = LLMChain(llm=llm, prompt=prompt_template)
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation"],
    allowed_tools=[tool.name for tool in tools]
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

#update extract_information function
def extract_information(entity, custom_prompt, backend_prompt, search_results):
    try:
        #Add a user prompt to clarify the task for the model
        user_prompt = f"Extract information for {entity} based on: {custom_prompt}. Search results: {search_results}."
        
        #Invoke the agent with a list of messages, including the user prompt and backend prompt
        response = agent_executor.invoke({"input": user_prompt, "entity": entity, "custom_prompt": custom_prompt, "backend_prompt": backend_prompt, "search_results": search_results, 
                                        "intermediate_steps": [],  #Include intermediate steps to track the agent's thought process
                                        "agent_scratchpad": ""    #Include agent scratchpad to track the agent's reasoning
                                        })
        #Assuming the response contain the extracted information in  a field named "output"
        return response.get("output", "Information not found.")
    except Exception as e:
        st.error(f"Error in LLM processing for {entity}: {e}")
        return "Information not found."

def upadate_gsheet(sheet_id, sheet_range, results_data):
    creds = get_gsheets_credentials()
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    
    #Clear existing data
    sheet.values().clear(spreadsheetId=sheet_id, range=sheet_range).execute()
    
    #Prepare data for update (header row + data rows)
    header = list(results_data[0].keys())
    data_rows = [list(row.values()) for row in results_data]
    values = [header] + data_rows
    
    #update the sheet
    body = {'values': values}
    result = sheet.values().update(spreadsheetId=sheet_id, range=sheet_range, valueInputOption="RAW", body=body).execute()
    
    print(f"{result.get('updatedCells')} cells updated")
    

def main():
    #.. title intro and text will be here
    st.title("SheetsAI")
    st.write("This app extracts information from web searches based on data from a CSV or Google Sheets file.")
    data_source = st.radio("Choose data source", ("Google Sheets", "CSV File"))
    uploaded_file = None
    df = None
    
    if data_source == "CSV File": 
        #File uploader
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            #column selection
            selected_columns = st.selectbox("Select columns to display", df.columns)
        
            #Data Preview
            st.subheader("Data Preview")
            st.dataframe(df)
        
            #custom prompt input
            st.subheader("Custom Search Prompt")
            custom_prompt = st.text_input("Enter your custom prompt (use placeholders like {company}):", "Get the email address of {company}")
        
            #backend prompt input
            st.subheader("Backend LLM Prompt")
            backend_prompt = st.text_input("Enter the prompt for the LLM:", "Extract the email address from the following text.")
        
        #rest of the code will be here
    elif data_source == "Google Sheets":
        sheet_id = st.text_input("Enter Google Sheets ID")
        sheet_range = st.text_input("Enter Google Sheets Range (eg - Sheet1!A:B)", value="Sheet1!A:B")
        
        if st.button("Connect to Google Sheets"):
            df = read_gsheet(sheet_id, sheet_range)
            if df is not None:
                #Column selection
                selected_columns = st.selectbox("Select columns to display", df.columns)
                
                #Data Preview
                st.subheader("Data Preview")
                st.dataframe(df)
                
                #Backend Prompt Input
                st.subheader("Backend LLM Prompt")
                backend_prompt = st.text_input("Enter the prompt for the LLM:", "Extract the email address from the following text.")
            else:
                st.error("Failed to connect to Google Sheets. Please check your ID and range.")
    
    if uploaded_file is not None or (data_source == "Google Sheets" and df is not None):
        if st.button("Start Extraction"):
            results_data = []
            for index, row in df.iterrows():
                entity = row[selected_columns]
                #perform web search(using your function)
                search_results = perform_web_search(entity, custom_prompt)
                
                #Extract information using the agent
                extracted_info = extract_information(entity, custom_prompt, backend_prompt, search_results)
                
                results_data.append({"Entity": entity, "Extracted Information": extracted_info})
            
            st.subheader("Extraction Results")
            st.dataframe(pd.DataFrame(results_data))
            
            st.download_button(
                label="Download Results as CSV",
                data = pd.DataFrame(results_data).to_csv(index=False).encode("utf-8"),
                file_name="extracted_results.csv",
                mime="text/csv"
            )
            
            if data_source == "Google Sheets" and 'sheet_id' in locals() and 'sheet_range' in locals():
                if st.button("Update Google Sheets"):
                    try:
                        upadate_gsheet(sheet_id, sheet_range, results_data) #Call the update function
                        st.success("Google Sheets updated successfully.")
                    except Exception as e:
                        st.error(f"Error updating Google Sheets: {e}")
    
if __name__ == "__main__":
    main()
