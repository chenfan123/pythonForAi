import os 
import requests
from dotenv import load_dotenv
from python_for_ai import get_llm_response, print_llm_response

load_dotenv('.env', override=True)  # 加载.env文件,override=True表示如果.env文件中有变量,则覆盖原来的变量

api_key = os.getenv("weather_api_key")

lat = 37.4419  # 纬度
lon = -122.1430  # 经度

url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}"

response = requests.get(url)
print(response.json())