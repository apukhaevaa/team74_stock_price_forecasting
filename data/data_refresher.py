import time
from schedule import Scheduler
from data_loader import load_data

data_schedule = Scheduler()
data_schedule.every().day.at('16:25').do(load_data)

while True:

    data_schedule.run_pending()
    time.sleep(1)

