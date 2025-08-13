from dotenv import load_dotenv
import os

load_dotenv()

def get_llm(model_name: str):
    if model_name == "gpt-4o" or model_name == "gpt-4o-mini":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=0.1,
            logprobs=True,
            top_logprobs=20,
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif model_name == "o3" or model_name == "o3-mini":
        if model_name == "o3": model_name = "o3-2025-04-16"
        else: model_name = "o3-mini-2025-01-31"
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            reasoning_effort="medium",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif model_name == "gemini-pro":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="models/gemini-1.5-pro",
            temperature=0.1,
            api_key=os.environ.get("GOOGLE_API_KEY", "")
        )
    elif model_name == "claude-3.5":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.1,
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
    elif model_name == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.1,
            logprobs=True,
            top_logprobs=20,
            api_key=os.environ.get("DEEPSEEK_API_KEY", "")
        )
    elif model_name == "deepseek-R1":
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            model="deepseek-reasoner",
            reasoning_effort="medium",
            api_key=os.environ.get("DEEPSEEK_API_KEY", "")
        )
    else:
        raise ValueError(f"Model {model_name} is not supported.")
