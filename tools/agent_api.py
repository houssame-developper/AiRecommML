from multiprocessing.connection import answer_challenge
from dotenv import load_dotenv
import os 
from langchain_groq import ChatGroq
from typing import TypedDict,Annotated,List
from pydantic import BaseModel
from langgraph.graph import add_messages
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.graph import StateGraph , START,END
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
sys_prompt = """You are an academic guidance agent specialized in Master’s degree orientation in Morocco.

Your task is to help Bachelor/Licence students who are unsure which Master specialization fits them best.

Language rule: Always respond in the same language used by the user, regardless of the language.

Your responsibilities:

Analyze the student’s Licence background based on studied courses and skills, not just the degree title.

Propose specific Master’s specializations available in Morocco, ranked by suitability.

For each suggested specialization, clearly state:

Why it matches the student’s academic background

Its relevance to the job market or academic progression

If needed, mention brief knowledge gaps and how the student can address them.

Constraints:

Focus only on Master-level specializations (not general fields).

Be realistic and concise.

Do not guarantee admission."""



llm_3_3 = ChatGroq(model="llama-3.3-70b-versatile",
               max_tokens=800,
               temperature=0.3,
               api_key=GROQ_API_KEY,
               verbose=False)

class RecommandationState(TypedDict):
  answer:str
  recomm:str
  data:dict

def generate_recommandation(state:RecommandationState):
 prompt = f"""
You are an educational advisor.

Give ONLY a direct explanation.
No introductions, no filler, no questions.

Explain why "{state["recomm"]}" fits the results {state["data"]}, and briefly mention one other suitable domain.
"""

   
 msg = [SystemMessage(content=sys_prompt),HumanMessage(content=prompt)]
 response = llm_3_3.invoke(msg)
 return {"answer":response}

builder = StateGraph(RecommandationState)
builder.add_node(generate_recommandation,"generate_recommandation")

builder.add_edge(START,"generate_recommandation")
builder.add_edge("generate_recommandation",END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

def agent_call(data,recomm):
    config = {"configurable":{"thread_id":str(uuid4())}}
    for chunk in graph.stream({"recomm":recomm,"data":data},config=config,stream_mode="values"):
        if "answer" in chunk:
            yield chunk["answer"].content