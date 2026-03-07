import logging
import os
from datetime import datetime
from typing import Any

import scrapy

from src.parsers.hodinky_parser import HodinkyParser
from src.items import BaseWatchItem
from src.items import WatchItem


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
        self.product_collection_pages_limit = 1

        self.crawled_pages = 0
        self.crawled_products = 0

        self.products_ignore_set = set()

        self.hodinky_parser = HodinkyParser()

    def parse(self, response):
        # self.crawler.stop()

        product_urls = self.hodinky_parser.parse_product_urls(response)
        for product_url in product_urls:
            if self.crawled_products >= self.products_per_page_limit:
                self.crawled_products = 0
                break

            if product_url not in self.products_ignore_set:
                self.crawled_products = self.crawled_products + 1
                self.products_ignore_set.add(product_url)
                self.logger.info("Submitting url {}".format(product_url))
                yield scrapy.Request(
                    product_url,
                    callback=self.parse_watch_product,
                    priority=10
                )

        self.crawled_pages = self.crawled_pages + 1
        if (self.hodinky_parser.has_next_page(response) and
            self.crawled_pages < self.product_collection_pages_limit):
            self.logger.info("Going to next page")
            yield scrapy.Request(
                self.hodinky_parser.next_page_url(),
                callback=self.parse,
            )

    def parse_watch_product(self, response):
        self.logger.info("Parsing product: {}".format(response.url))
        # TODO: here limit by ignore set and pages count
        item = WatchItem()
        self.hodinky_parser.parse_watch_item(item, response)
        variant_urls = self.hodinky_parser.parse_variant_urls(response)
        if len(variant_urls) == 0:
            yield item
            return

        item["_variants_remaining"] = len(variant_urls)
        for url in variant_urls:
            self.logger.info("Submitting variant request {}".format(hash(url)))
            self.products_ignore_set.add(url)
            yield scrapy.Request(
                url,
                callback=self.parse_variant,
                meta={"item": item},
                priority=100
            )

    def parse_variant(self, response):
        self.logger.info("Parsing variant: {}".format(hash(response.url)))
        watches_item = response.meta["item"]
        variant_item = BaseWatchItem()

        self.hodinky_parser.parse_base_watch_item(variant_item, response)
        watches_item.setdefault("variants", []).append(variant_item)
        watches_item["_variants_remaining"] -= 1
        if watches_item["_variants_remaining"] == 0:
            del watches_item["_variants_remaining"]
            yield watches_item