import logging 
import os 
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
logs_dir = os.path.join(os.getcwd(), 'logs')   #  directory only, no filename
os.makedirs(logs_dir, exist_ok=True)            # creates the folder correctly

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)  #  now a proper file path

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)