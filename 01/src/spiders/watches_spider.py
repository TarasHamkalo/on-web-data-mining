import regex
import scrapy
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

    products_per_page_limit = 2
    product_collection_pages_limit = 2

    crawled_pages = 0
    crawled_products = 0

    def parse(self, response):
        product_urls = response.css("#productList > div > div > a::attr(href)").getall()
        for product_url in product_urls:
            if self.crawled_products >= self.products_per_page_limit:
                self.crawled_products = 0
                break

            self.crawled_products = self.crawled_products + 1
            normalized_url = response.urljoin(product_url)

            yield response.follow(normalized_url, callback=self.parse_watch_product)

        is_next_page_available = response.xpath("//*[@aria-label='Následující stránka']")

        self.crawled_pages = self.crawled_pages + 1
        if is_next_page_available is not None and self.crawled_pages < self.product_collection_pages_limit:
            yield scrapy.Request(
                self.pagination_format + str(self.crawled_pages + 1),
                callback=self.parse,
            )

    def parse_watch_product(self, response):
        item = WatchItem()
        item["url"] = response.url
        item["title"] = response.css(".product-buybox h1::text").get()

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
        item["strap_material"] = self.extract_param(response, "Materiál řemínku")
        item["strap_color"] = self.extract_param(response, "Barva řemínku/tahu")
        item["dial_color"] = self.extract_param(response, "Barva ciferníku")

        item["functions"] = self.extract_param(params_body, "Funkce hodinek")
        if item["functions"]:
            item["functions"] = [s.strip() for s in str(item["functions"]).split(";")]

        yield item

    def extract_param(self, params_body, param):
        parameter_value = params_body.xpath(
            f"//*[contains(text(), '{param}')]/following-sibling::*/text()"
        )

        if parameter_value:
            return parameter_value.get().strip()

        return None

