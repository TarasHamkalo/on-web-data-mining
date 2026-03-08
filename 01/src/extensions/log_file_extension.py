import os
import datetime
import logging
from scrapy import signals

# source:https://dev.to/ajitkumar/dynamic-log-file-for-each-spiders-scrapy-logging-11a0
class LogFileExtension:
    @classmethod
    def from_crawler(cls, crawler):
        ext = cls()
        crawler.signals.connect(ext.spider_opened, signal=signals.spider_opened)
        return ext

    def spider_opened(self, spider):
        # Create a logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # Generate a dynamic log file name
        log_file_name =\
            f"{log_dir}/{spider.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

        # Set up logging to the file
        file_handler = logging.FileHandler(log_file_name, mode="w")
        file_handler.setLevel(logging.INFO)  # Adjust level as needed
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )

        # Get Scrapy's root logger and add the file handler
        logger = logging.getLogger()
        logger.addHandler(file_handler)

        spider.log_file = log_file_name
        # Debugging message
        print(f"Log file for spider {spider.name} set to: {log_file_name}")