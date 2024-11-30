import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    trim_messages,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
import json
from typing import Optional

app = FastAPI()

@tool
def get_user_age(name: str) -> str:
    """Use this tool to find the user's age."""
    if "bob" in name.lower():
        return "42 years old"
    return "41 years old"

memory = MemorySaver()
model = ChatOpenAI(streaming=True)

def state_modifier(state) -> list[BaseMessage]:
    return trim_messages(
        state["messages"],
        token_counter=len,
        max_tokens=16000,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )

agent = create_react_agent(
    model,
    tools=[get_user_age],
    checkpointer=memory,
    state_modifier=state_modifier,
)

class ChatInput(BaseModel):
    message: str
    thread_id: Optional[str] = None  # Optional thread_id field

@app.post("/chat")
async def chat(input_data: ChatInput):
    thread_id = input_data.thread_id or uuid.uuid4()  # Use provided thread_id or generate a new one
    config = {"configurable": {"thread_id": thread_id}}
    
    input_message = HumanMessage(content=input_data.message)
    
    async def generate():
        async for event in agent.astream_events(
            {"messages": [input_message]}, 
            config,
            version="v2"
        ):
            kind = event["event"]
            
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
            
            elif kind == "on_tool_start":
                tool_input = str(event['data'].get('input', ''))
                yield f"data: {json.dumps({'type': 'tool_start', 'tool': event['name'], 'input': tool_input})}\n\n"
            
            elif kind == "on_tool_end":
                tool_output = str(event['data'].get('output', ''))
                yield f"data: {json.dumps({'type': 'tool_end', 'tool': event['name'], 'output': tool_output})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 