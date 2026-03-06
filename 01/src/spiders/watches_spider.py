import regex
import scrapy

from src.items import BaseWatchItem
from src.items import WatchItem


class WatchesSpider(scrapy.Spider):
    name = "watches"
    allowed_domains = [
        "www.hodinky.cz",
        "www.helveti.cz",
        "www.hodinky-koscom.cz"
    ]

    pagination_format = "https://www.hodinky.cz/citizen/?page="

    start_urls = [
        "https://www.hodinky.cz/citizen/",
    ]

    products_per_page_limit = 5
    product_collection_pages_limit = 3

    crawled_pages = 0
    crawled_products = 0


    products_ignore_set = set()

    def parse(self, response):
        product_urls = response.css("#productList > div > div > a::attr(href)").getall()
        for product_url in product_urls:
            if self.crawled_products >= self.products_per_page_limit:
                self.crawled_products = 0
                break

            normalized_url = response.urljoin(product_url)
            if normalized_url not in self.products_ignore_set:
                # state has to be updated here, as it is not know when will parse watch be called
                self.crawled_products = self.crawled_products + 1
                self.products_ignore_set.add(normalized_url)
                yield response.follow(normalized_url, callback=self.parse_watch_product)

        is_next_page_available = response.xpath("//*[@aria-label='Následující stránka']")

        self.crawled_pages = self.crawled_pages + 1
        if is_next_page_available is not None and self.crawled_pages < self.product_collection_pages_limit:
            yield scrapy.Request(
                self.pagination_format + str(self.crawled_pages + 1),
                callback=self.parse,
            )

    def parse_watch_product(self, response):
        print(response.url)
        item = WatchItem()

        self.extract_base_watch_item(item, response)

        item["title"] = response.css(".product-buybox h1::text").get()
        item["brand"] = response.xpath(
            '//div[contains(@class,"product-buybox")]//h1/preceding-sibling::*[1]//text()'
        ).get()

        item["functions"] = self.extract_param(response, "Funkce hodinek")
        if item["functions"]:
            item["functions"] = [s.strip() for s in str(item["functions"]).split(";")]

        variant_urls = response.xpath(
            "//*[contains(text(), 'Barevná provedení')]/following-sibling::*//a/@href"
        ).getall()

        if not variant_urls:
            yield item
            return

        normalized_urls = []
        for variant_url in variant_urls:
            normalized_url = response.urljoin(variant_url)
            # current object url is already added to ignore set
            if normalized_url != response.url:
                normalized_urls.append(normalized_url)

        item["_variants_remaining"] = len(normalized_urls)
        for i, url in enumerate(normalized_urls):
            self.products_ignore_set.add(url)
            yield scrapy.Request(
                url,
                callback=self.parse_variant,
                meta={"item": item},
            )


    def parse_variant(self, response):
        watches_item = response.meta["item"]

        variant_item = BaseWatchItem()
        self.extract_base_watch_item(variant_item, response)
        watches_item.setdefault("variants", []).append(variant_item)

        watches_item["_variants_remaining"] -= 1
        print(watches_item["url"], ">>>>", watches_item["_variants_remaining"])
        if watches_item["_variants_remaining"] == 0:
            del watches_item["_variants_remaining"]
            yield watches_item

    def extract_base_watch_item(self, item, response):
        item["url"] = response.url

        price_text = response.css(".product-buybox .h2-no-caps > span::text").get().strip()
        item["currency"] = regex.search(r"\p{L}+", price_text).group()
        item["price"] = int(regex.sub(r"\D", "", price_text))

        item["image_url"] = response.css(
            'meta[name="og:image"]::attr(content)'
        ).get()

        params_body = response.xpath(
            "//*[contains(text(), 'Základní parametry')]/parent::*/following-sibling::*"
        )
        item["case_material"] = self.extract_param(params_body, "Materiál pouzdra")
        item["strap_material"] = self.extract_param(params_body, "Materiál řemínku")
        item["strap_color"] = self.extract_param(params_body, "Barva řemínku/tahu")
        item["dial_color"] = self.extract_param(params_body, "Barva ciferníku")

    def extract_param(self, params_body, param):
        parameter_value = params_body.xpath(
            f"//*[contains(text(), '{param}')]/following-sibling::*/text()"
        )

        if parameter_value:
            return parameter_value.get().strip()

        return None

