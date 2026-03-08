from collections import deque
from typing import Any

import scrapy
from scrapy.http.response import Response

from src.items import WatchItem
from src.parsers.hodinky_parser import HodinkyParser


class CategoryContext:
    """
    Holds crawler state for a single start_url (category).
    """

    def __init__(self, start_url):
        self.crawled_pages = 0
        self.crawled_products = 0
        self.product_url_queue = deque()
        self.has_next_page = False
        self.stop_serving_new_requests = False

        self.start_url = start_url

        self.hodinky_parser = HodinkyParser(start_url)

class WatchesSpider(scrapy.Spider):
    name = "watches"

    allowed_domains = [
        "www.hodinky.cz",
    ]

    start_urls = [
        "https://www.hodinky.cz/citizen",
        "https://www.hodinky.cz/casio",
        "https://www.hodinky.cz/seiko/"
        # "https://www.hodinky.cz/hodinky/designove-hodinky/automat-natahovani-pohybem-ruky"
    ]

    def __init__(self, name: str | None = None, **kwargs: Any):
        super().__init__(name, **kwargs)

        self.products_per_page_limit = 2
        self.product_collection_pages_limit = 2
        self.products_ignore_set = set()

    def start_requests(self):
        """
        Initialize crawling context per category.
        """
        for url in self.start_urls:
            ctx = CategoryContext(url)
            yield scrapy.Request(
                url,
                callback=self.parse,
                meta={"ctx": ctx},
            )

    def parse(self, response: Response):
        """
        Extract product URLs and populate the product queue.
        Separate not scrapy related queue used to limit amount of concurrent request and
        simplify deduplication of products.

        :param response: scrapy response object for watches collection page
        :return: yields single request to new product
        """

        ctx: CategoryContext = response.meta["ctx"]
        product_urls = ctx.hodinky_parser.parse_product_urls(response)

        for product_url in product_urls:
            ctx.product_url_queue.append(product_url)

        self.logger.info(
            f"[{ctx.start_url}] Product queue contains [{len(product_urls)}] products"
        )

        ctx.has_next_page = ctx.hodinky_parser.has_next_page(response)
        ctx.crawled_pages += 1
        yield from self.create_single_request(ctx)

    def parse_watch_product(self, response):
        """
        Parse product page, schedule variants and next product.
        """
        ctx: CategoryContext = response.meta["ctx"]
        is_variant = response.meta.get("is_variant", False)

        stats = self.crawler.stats

        if is_variant:
            self.logger.info(f"Parsing variant: {response.url}")
            stats.inc_value("variants_scraped")
        else:
            self.logger.info(f"Parsing product: {response.url}")
            stats.inc_value("products_scraped")

        if not is_variant:
            ctx.crawled_products += 1
            self.logger.info(
                f"Unique products parsed: {ctx.crawled_products}"
            )

        item = WatchItem()
        ctx.hodinky_parser.parse_watch_item(item, response)

        variant_urls = ctx.hodinky_parser.parse_variant_urls(response)
        item["_variant_urls"] = variant_urls

        yield item

        yield from self.create_product_variant_requests(ctx, variant_urls)

        if not is_variant:
            yield from self.create_single_request(ctx)

    def create_single_request(self, ctx: CategoryContext):
        """
        Creates single request either to download new product or collection of products.
        """
        if ctx.stop_serving_new_requests:
            return

        if (
            ctx.crawled_products >= self.products_per_page_limit
            or not ctx.product_url_queue
        ):
            self.logger.info("Products per page limit reached or product queue is empty")
            yield from self.create_next_page_request(ctx)
            return

        while ctx.product_url_queue:
            product_url = ctx.product_url_queue.popleft()

            if product_url in self.products_ignore_set:
                continue

            self.products_ignore_set.add(product_url)
            self.logger.info(f"Creating product request: {product_url}")
            yield scrapy.Request(
                product_url,
                callback=self.parse_watch_product,
                meta={
                    "is_variant": False,
                    "ctx": ctx,
                },
            )

            break

    def create_next_page_request(self, ctx: CategoryContext):
        """
        Creates request to download next page of products, in case that there are
        pages on web and limit was not hit.
        """

        if ctx.crawled_pages < self.product_collection_pages_limit:
            ctx.crawled_products = 0
            ctx.product_url_queue = deque()
            next_page = ctx.hodinky_parser.next_page_url()

            self.logger.info(
                f"Going to next page [{ctx.crawled_pages + 1}] -> {next_page}"
            )
            yield scrapy.Request(
                next_page,
                callback=self.parse,
                meta={"ctx": ctx},
            )
            return

        self.logger.info("Page limit reached, stopping category crawl")
        ctx.stop_serving_new_requests = True

    def create_product_variant_requests(self, ctx: CategoryContext, variant_urls):
        """
        Create variant product requests.
        """

        for url in variant_urls:
            if url in self.products_ignore_set:
                continue

            self.products_ignore_set.add(url)
            self.logger.info(f"Creating variant request: {url}")
            yield scrapy.Request(
                url,
                callback=self.parse_watch_product,
                meta={
                    "is_variant": True,
                    "ctx": ctx,
                },
                priority=2,
            )
