import logging
import os
from datetime import datetime

from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


class VariantMergingPipeline():

    def __init__(self):
        self.seen_items = {}
        self.logger = None

    def open_spider(self, spider):
        # Create log directory
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        log_file = f"{log_dir}/variant_pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

        self.logger = logging.getLogger("variant_pipeline")
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file, mode="w")
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )

        self.logger.addHandler(handler)

        spider.variant_pipeline_log = log_file

        self.logger.info("VariantMergingPipeline started")

    def process_item(self, item, spider):
        """
        Caches given item (by raising drop item exception) till all it's variants are
        merged
        """
        adapter = ItemAdapter(item)
        url = adapter.get("url")
        variant_urls = adapter.get("_variant_urls", [])
        self.logger.info(f"Processing item {url}")
        if len(variant_urls) == 0:
            self.logger.info(f"Item has no variants, passing through {url}")
            del adapter["_variant_urls"]
            return item

        for variant_url in variant_urls:
            if variant_url in self.seen_items:
                self.logger.info(f"Found variant match: {url} -> {variant_url}")

                del adapter["_variant_urls"]
                original = self.seen_items[variant_url]
                original.setdefault("variants", []).append(dict(adapter))
                original["_variant_urls"].remove(url)

                if not original["_variant_urls"]:
                    self.logger.info(f"All variants merged for {original['url']}")
                    self.seen_items.pop(variant_url, None)
                    del original["_variant_urls"]
                    return original

                raise DropItem("Item merged as variant")

        self.logger.info(f"Caching base item {url}")
        self.seen_items[url] = item
        raise DropItem("Item cached waiting for variants")

    def close_spider(self, spider):
        self.logger.info("VariantMergingPipeline finished")