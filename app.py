import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
import uvicorn

# ==========================================
# 1. LLM SETUP
# ==========================================
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

llm_flash = ChatGoogleGenerativeAI(
    model="models/gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
    temperature=0
)

# ==========================================
# 2. STATE DEFINITION
# ==========================================
class CrewState(TypedDict):
    messages: List[BaseMessage]
    code: Optional[str]
    report: Optional[str]
    status: Optional[str]

# ==========================================
# 3. TOOLS
# ==========================================
@tool
def run_python_code(code: str) -> str:
    """Execute python code and return output."""
    if not isinstance(code, str):
        code = str(code)
    clean_code = code.replace('```python', '').replace('```', '').strip()
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    try:
        exec(clean_code, {}, {})
        result = new_stdout.getvalue()
    except Exception as e:
        result = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
    return result.strip() if result.strip() else "Success (no output)"

@tool
def generate_test_cases(task_description: str) -> str:
    """Generate test scenarios for a coding task."""
    prompt = f"Generate 3-5 test scenarios for: {task_description}"
    response = llm_flash.invoke(prompt)
    
    content = response.content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return " ".join(text_parts) if text_parts else str(content)
    return str(content)

# ==========================================
# 4. GRAPH NODES
# ==========================================
def developer_node(state: CrewState):
    task = state['messages'][-1].content
    prompt = f"Write Python code for: {task}. Only return code, no explanation."
    response = llm_flash.invoke(prompt)
    
    content = response.content
    if isinstance(content, list):
        code_str = ""
        for block in content:
            if isinstance(block, dict) and "text" in block:
                code_str += block["text"]
            elif isinstance(block, str):
                code_str += block
    else:
        code_str = str(content)
    
    return {"code": code_str, "status": "code_written"}

def tester_node(state: CrewState):
    task = state['messages'][-1].content
    tests = generate_test_cases.invoke({"task_description": task})
    result = run_python_code.invoke({"code": state['code']})
    report = f"Output:\n{result}\n\nTests:\n{tests}"
    return {"report": report, "status": "tests_completed"}

def final_node(state: CrewState):
    # Create a response message with the report
    response_msg = HumanMessage(content=state['report'])
    return {"messages": [response_msg], "status": "done"}

# ==========================================
# 5. BUILD GRAPH
# ==========================================
workflow = StateGraph(CrewState)
workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)
workflow.add_node("final", final_node)

workflow.set_entry_point("developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "final")
workflow.add_edge("final", END)

graph = workflow.compile()

# ==========================================
# 6. FASTAPI APP
# ==========================================
app = FastAPI(title="LangGraph Coding Agent")

class AgentInput(BaseModel):
    input: str = Field(description="Your coding task")

def format_input(x):
    return {"messages": [HumanMessage(content=x["input"])]}

# ✅ FIXED: Extract response from state
def extract_response(state):
    """Extract the final answer from the state."""
    # Check if there's a report
    if state.get("report"):
        return state["report"]
    
    # Check messages
    if state.get("messages") and len(state["messages"]) > 0:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "content"):
            return last_msg.content
    
    # Check if there's code
    if state.get("code"):
        return f"Code generated:\n{state['code']}"
    
    return "No response generated"

chain = RunnableLambda(format_input) | graph | RunnableLambda(extract_response)
chain = chain.with_types(input_type=AgentInput, output_type=str)

add_routes(app, chain, path="/agent")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

print("✅ app.py created with fixed extract_response!")
