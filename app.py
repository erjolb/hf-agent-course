from smolagents import CodeAgent,DuckDuckGoSearchTool,load_tool,tool,LiteLLMModel,VisitWebpageTool
import datetime
import requests
import pytz
import yaml
import os
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def get_asn_holder(asn:str)-> str:
    """Gets ASN holder information. ALWAYS wrap result with final_answer tool.
    Args:
        asn: A string representing the ASN (e.g., 'AS15169').
    Returns:
        str: ASN information that MUST be wrapped with final_answer
    """
    try:
        response = requests.get(f"https://stat.ripe.net/data/as-overview/data.json?resource={asn}")
        if response.status_code == 200:
            data = response.json()
            holder = data['data']
            return f"Holder of {asn} is {holder['holder']}"
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """Gets current time in timezone. ALWAYS wrap result with final_answer tool.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    Returns:
        str: time information that MUST be wrapped with final_answer
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error: {str(e)}"


final_answer = FinalAnswerTool()
web_search = DuckDuckGoSearchTool()
visit_webpage = VisitWebpageTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

api_key = os.environ['DEEPSEEK_API_KEY']
model = LiteLLMModel(
    model_id="deepseek/deepseek-chat", # This model is a bit weak for agentic behaviours though
    api_key=api_key,
    )


# Import tool from Hub
# image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_current_time_in_timezone, get_asn_holder, web_search, visit_webpage], ## add your tools here (don't remove final answer)
    max_steps=5,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()