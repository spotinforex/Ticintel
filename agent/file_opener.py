from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

def open_file(filename):
    '''Function to open system prompts '''
    try: 
        root_path = Path.cwd()
        full_path = f"{root_path}/agent/system_prompt/{filename}"
        with open(full_path, "r") as f:
            content = f.read()
        if content:
            return content
        else:
            logging.warning(f"No content was found in file: {filename}")
            raise 
    except Exception as e:
        logging.error(f"An Error Occurred when opening system prompt. Error: {e}")
        raise

