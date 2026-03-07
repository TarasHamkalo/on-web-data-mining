from queue import Queue
from typing import Any

import scrapy
from scrapy.http.response import Response

from src.items import WatchItem
from src.parsers.hodinky_parser import HodinkyParser


class WatchesSpider(scrapy.Spider):
    name = "watches"
    allowed_domains = [
        "www.hodinky.cz",
        "www.helveti.cz",
        "www.hodinky-koscom.cz"
    ]

    start_urls = [
        "https://www.hodinky.cz/citizen/",
    ]


    def __init__(self, name: str | None = None, **kwargs: Any):
        super().__init__(name, **kwargs)

        self.products_per_page_limit = 2
        self.product_collection_pages_limit = 2

        self.crawled_pages = 0
        self.crawled_products = 0

        self.products_ignore_set = set()

        self.hodinky_parser = HodinkyParser()

        self.product_url_queue = Queue()

        self.has_next_page = False

        self.stop_serving_new_requests = False

    def parse(self, response: Response) -> any:
        product_urls = self.hodinky_parser.parse_product_urls(response)
        for product_url in product_urls:
            self.product_url_queue.put(product_url)

        self.logger.info(f"Product queue contains [{len(product_urls)}] products")

        self.has_next_page = self.hodinky_parser.has_next_page(response)
        self.crawled_pages = self.crawled_pages + 1

        yield from self.create_single_request()

    def parse_watch_product(self, response):
        is_variant = response.meta.get("is_variant")
        if not is_variant:
            self.crawled_products += 1
            self.logger.info("Unique products parsed: {}".format(self.crawled_products))
            self.logger.info("Parsing product: {}".format(response.url))

        if is_variant:
            self.logger.info("Parsing product variant: {}".format(response.url))

        item = WatchItem()
        self.hodinky_parser.parse_watch_item(item, response)
        variant_urls = self.hodinky_parser.parse_variant_urls(response)
        item["_variant_urls"] = variant_urls
        yield item

        yield from self.create_product_variant_requests(variant_urls)

        if not is_variant:
            yield from self.create_single_request()

    def create_single_request(self):
        if self.stop_serving_new_requests:
            return

        if (self.crawled_products >= self.products_per_page_limit or
          self.product_url_queue.empty()):
            self.logger.info("Products per page limit reached or product queue is empty")
            yield from self.create_next_page_request()
            return

        while not self.product_url_queue.empty():
            product_url = self.product_url_queue.get()
            if product_url in self.products_ignore_set:
                continue

            self.products_ignore_set.add(product_url)
            self.logger.info("Creating product request: {}".format(product_url))
            yield scrapy.Request(
                product_url,
                callback=self.parse_watch_product,
                meta={"is_variant": False},
            )

            break

        return

    def create_next_page_request(self):
        if self.crawled_pages < self.product_collection_pages_limit:
            self.crawled_products = 0
            self.product_url_queue = Queue() #TODO: init new one? or clean?
            self.logger.info("Going to next page [{}]".format(self.crawled_pages + 1))
            yield scrapy.Request(
                self.hodinky_parser.next_page_url(),
                callback=self.parse,
            )
            return

        self.logger.info("Pages limit reached no new pages gonna be crawled")
        self.stop_serving_new_requests = True


    def create_product_variant_requests(self, variant_urls):
        for url in variant_urls:
            if url in self.products_ignore_set:
                continue

            self.logger.info("Creating product variant request {}".format(url))
            self.products_ignore_set.add(url)
            yield scrapy.Request(
                url,
                callback=self.parse_watch_product,
                meta={"is_variant": True},
                priority=2
            )
