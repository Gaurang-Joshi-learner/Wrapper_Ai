import os
import time
import json
import asyncio
import aiohttp
import logging
from dotenv import load_dotenv

# ---------------------------
#  Setup
# ---------------------------
load_dotenv()

# Load API Keys from .env
FALCONAI_API_KEY = os.getenv("FALCONAI_API_KEY")
QWENAI_API_KEY = os.getenv("QWENAI_API_KEY")
DEEPSEEKAI_API_KEY = os.getenv("DEEPSEEKAI_API_KEY")

# Logging
logging.basicConfig(filename="wrapper.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


#  Falcon (AI71 Platform)
# ---------------------------
async def query_falcon(prompt: str, api_key: str = FALCONAI_API_KEY, temperature: float = 0.7, max_tokens: int = 500):
    if not api_key:
        return {"model": "Falcon-7B (AI71)", "response": "⚠️ Missing API key in .env", "tokens": 0, "time_ms": 0}

    url = "https://api.ai71.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "falcon-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=60) as response:
                data = await response.json()
                end_time = time.time()
                elapsed = int((end_time - start_time) * 1000)

                # Parse
                if "choices" in data and data["choices"]:
                    text = data["choices"][0]["message"]["content"]
                elif "detail" in data:
                    text = data["detail"]
                else:
                    text = json.dumps(data)

                logging.info(f"[Falcon] Prompt: {prompt[:40]}... | Time: {elapsed}ms")
                return {"model": "Falcon-7B (AI71)", "response": text.strip(), "tokens": 0, "time_ms": elapsed}

    except Exception as e:
        logging.error(f"[Falcon] Error: {e}")
        return {"model": "Falcon-7B (AI71)", "response": f"Error: {str(e)}", "tokens": 0, "time_ms": 0}


# ---------------------------
#  Qwen (via OpenRouter)

async def query_qwen(prompt: str, api_key: str = QWENAI_API_KEY, temperature: float = 0.7, max_tokens: int = 500):
    if not api_key:
        return {"model": "Qwen-7B (OpenRouter)", "response": "⚠️ Missing API key in .env", "tokens": 0, "time_ms": 0}

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Wrapper.ai",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "qwen/qwen-2.5-72b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=60) as response:
                data = await response.json()
                end_time = time.time()
                elapsed = int((end_time - start_time) * 1000)

                if "choices" in data and data["choices"]:
                    text = data["choices"][0]["message"]["content"]
                elif "error" in data:
                    text = data["error"].get("message", str(data))
                else:
                    text = str(data)

                logging.info(f"[Qwen] Prompt: {prompt[:40]}... | Time: {elapsed}ms")
                return {"model": "Qwen-7B (OpenRouter)", "response": text.strip(), "tokens": 0, "time_ms": elapsed}

    except Exception as e:
        logging.error(f"[Qwen] Error: {e}")
        return {"model": "Qwen-7B (OpenRouter)", "response": f"Error: {str(e)}", "tokens": 0, "time_ms": 0}


# ---------------------------
#  DeepSeek (Official API)
# ---------------------------
async def query_deepseek(prompt: str, api_key: str = DEEPSEEKAI_API_KEY, temperature: float = 0.7, max_tokens: int = 500):
    if not api_key:
        return {"model": "DeepSeek (API)", "response": "⚠️ Missing API key in .env", "tokens": 0, "time_ms": 0}

    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=60) as response:
                data = await response.json()
                end_time = time.time()
                elapsed = int((end_time - start_time) * 1000)

                if "choices" in data and data["choices"]:
                    text = data["choices"][0]["message"]["content"]
                elif "error" in data:
                    text = data["error"].get("message", str(data))
                else:
                    text = str(data)

                logging.info(f"[DeepSeek] Prompt: {prompt[:40]}... | Time: {elapsed}ms")
                return {"model": "DeepSeek (API)", "response": text.strip(), "tokens": 0, "time_ms": elapsed}

    except Exception as e:
        logging.error(f"[DeepSeek] Error: {e}")
        return {"model": "DeepSeek (API)", "response": f"Error: {str(e)}", "tokens": 0, "time_ms": 0}


# ---------------------------
#  Coordinator
# ---------------------------
async def query_all_llms(prompt: str, temperature: float = 0.7, max_tokens: int = 500):
    """
    Runs all available LLMs asynchronously.
    """
    tasks = [
        query_falcon(prompt,api_key=FALCONAI_API_KEY, temperature=temperature, max_tokens=max_tokens),
        query_qwen(prompt,api_key=QWENAI_API_KEY, temperature=temperature, max_tokens=max_tokens),
        query_deepseek(prompt,api_key=DEEPSEEKAI_API_KEY, temperature=temperature, max_tokens=max_tokens),
    ]

    results = await asyncio.gather(*tasks)
    return results
