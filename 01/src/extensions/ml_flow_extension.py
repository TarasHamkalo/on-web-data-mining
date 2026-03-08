import os

import mlflow
from datetime import datetime
from scrapy import signals
from src.spiders.watches_spider import WatchesSpider


class MLflowExtension:

    def __init__(self):
        self.start_time = None

    @classmethod
    def from_crawler(cls, crawler):
        ext = cls()


        crawler.signals.connect(ext.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(ext.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(ext.feed_exporter_closed, signal=signals.feed_exporter_closed)

        # crawler.signals.connect(ext.spider_opened, signal=crawler.signals.spider_opened)
        # crawler.signals.connect(ext.spider_closed, signal=crawler.signals.spider_closed)
        return ext

    def spider_opened(self, spider):
        if not isinstance(spider, WatchesSpider):
            return

        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("hw-01-web-crawler")

        self.start_time = datetime.now()

        mlflow.start_run()

        mlflow.log_param("domains", ",".join(spider.allowed_domains))

        mlflow.log_param("spider_name", spider.name)
        mlflow.log_param("start_urls", ",".join(spider.start_urls))
        mlflow.log_param("start_time", self.start_time.isoformat() + "Z")
        mlflow.log_param("products_per_page_limit", spider.products_per_page_limit)
        mlflow.log_param("collection_page_limit", spider.product_collection_pages_limit)


        settings = spider.crawler.settings
        # scrapy settings
        mlflow.log_param(
            "concurrent_requests",
            settings.getint("CONCURRENT_REQUESTS")
        )

        mlflow.log_param(
            "download_delay",
            settings.getfloat("DOWNLOAD_DELAY")
        )

        mlflow.log_param(
            "concurrent_requests_per_domain",
            settings.getint("CONCURRENT_REQUESTS_PER_DOMAIN")
        )

    def spider_closed(self, spider):
        if not isinstance(spider, WatchesSpider):
            return

        end_time = datetime.now()

        duration = (end_time - self.start_time).total_seconds()

        stats = spider.crawler.stats.get_stats()

        mlflow.log_metric("products_scraped", stats.get("products_scraped", 0))
        mlflow.log_metric("variants_scraped", stats.get("variants_scraped", 0))

        mlflow.log_metric("total_items", stats.get("item_scraped_count", 0))

        mlflow.log_metric("total_requests", stats.get("downloader/request_count", 0))
        mlflow.log_metric("total_responses", stats.get("downloader/response_count", 0))

        mlflow.log_metric("crawl_duration_seconds", duration)

        mlflow.log_artifact("extracted_data.json", artifact_path="data")
        spider_log_file = getattr(spider, "log_file", None)
        if spider_log_file and os.path.exists(spider_log_file):
            mlflow.log_artifact(spider_log_file, artifact_path="logs")

        pipeline_log_file = getattr(spider, "variant_pipeline_log", None)
        if pipeline_log_file and os.path.exists(pipeline_log_file):
                mlflow.log_artifact(pipeline_log_file, artifact_path="logs")

    def feed_exporter_closed(self):
        if os.path.exists("extracted_data.json"):
            mlflow.log_artifact("extracted_data.json", artifact_path="data")

        mlflow.end_run()


