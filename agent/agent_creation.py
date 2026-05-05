from backboard import BackboardClient
from dotenv import load_dotenv 
from agent.file_opener import open_file 
import logging, os, asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

load_dotenv()

api_key = os.getenv("api_key")

client = BackboardClient(api_key=api_key)

async def create_search_assistant():
    try:
      assistant = await client.create_assistant(
          name = "Search Agent",
          system_prompt = open_file("search_system_prompt.txt"),
          tok_k = 20,
          )
      logging.info(f"Search assistant id: {assistant.assistant_id}")
    except Exception as e:
      logging.error(f"An error occurred when creating search assistant. Error: {e}")

async def create_extract_assistant():
    try:
      assistant = await client.create_assistant(
          name = "Extraction Agent",
          system_prompt = open_file("extraction_system_prompt.txt"),
          tok_k = 20,
          )
      logging.info(f"Extraction assistant id: {assistant.assistant_id}")
    except Exception as e:
      logging.error(f"An error occurred when creating extraction assistant. Error: {e}")

async def create_contradiction_assistant():
    try:
      assistant = await client.create_assistant(
          name = "Contradiction Agent",
          system_prompt = open_file("contradiction_system_prompt.txt"),
          tok_k = 20,
          )
      logging.info(f"Contradiction assistant id: {assistant.assistant_id}")
    except Exception as e:
      logging.error(f"An error occurred when creating contradiction assistant. Error: {e}")

async def create_synthesis_assistant():
    try:
      assistant = await client.create_assistant(
          name = "Synthesis Agent",
          system_prompt = open_file("synthesis_system_prompt.txt"),
          tok_k = 20,
          )
      logging.info(f"Synthesis assistant id: {assistant.assistant_id}")
    except Exception as e:
      logging.error(f"An error occurred when creating synthesis assistant. Error: {e}")

if __name__ == "__main__":
    asyncio.run(create_synthesis_assistant())
