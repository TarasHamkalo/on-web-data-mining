from typing import List

import regex
import requests
import scrapy


# Parser for watches item data over hodinky.cz
class HodinkyParser():
  base_watch_parameters_translation_map = {
    "case_material": "Materiál pouzdra",
    "strap_material": "Materiál řemínku",
    "strap_color": "Barva řemínku/tahu",
    "dial_color": "Barva ciferníku",
  }

  watch_parameters_translation_map = {
    "functions": "Funkce hodinek",
    "variants": "Barevná provedení"
  }

  pagination_format = "/?page="

  def __init__(self, base_url):
    self.current_page = 1
    self.base_url = base_url

  def has_next_page(self, response) -> bool:
    is_next_page_available = response.xpath("//*[@aria-label='Následující stránka']").get()
    return is_next_page_available is not None

  def next_page_url(self) -> str:
    self.current_page += 1
    return self.base_url + self.pagination_format + str(self.current_page)

  def parse_product_urls(self, response):
    """
    :param response: object containing html body of products page from hodinky.cz
    :return: set of normalized urls to particular products
    """
    product_urls = response.css("#productList > div > div > a::attr(href)").getall()
    normalized_urls = set()
    for product_url in product_urls:
      normalized_url = response.urljoin(product_url)
      normalized_urls.add(normalized_url)
    return normalized_urls

  def parse_watch_item(self, item, response) -> None:
    """
    :param item: scrapy item to fill fields into
    :param response: object containing html body of product page from hodinky.cz
    :return: none
    """
    self.parse_base_watch_item(item, response)

    item["title"] = response.css(".product-buybox h1::text").get()
    item["brand"] = response.xpath(
      '//*[contains(@class,"product-buybox")]//h1/preceding-sibling::*/text()'
    ).get()

    item["functions"] = self.parse_parameter(
      response, self.watch_parameters_translation_map["functions"]
    )

    if item["functions"]:
      item["functions"] = [s.strip() for s in str(item["functions"]).split(";")]


  def parse_base_watch_item(self, item, response) -> None:
    """
    :param item: scrapy item to fill fields
    :param response: object containing html body of product page from hodinky.cz
    """
    item["url"] = response.url

    price_text = response.css(".product-buybox .h2-no-caps > span::text").get().strip()
    item["currency"] = regex.search(r"\p{L}+", price_text).group()
    item["price"] = int(regex.sub(r"\D", "", price_text))

    item["image_url"] = response.css('meta[name="og:image"]::attr(content)').get()
    for parameter, parameter_translation in self.base_watch_parameters_translation_map.items():
      item[parameter] = self.parse_parameter(response, parameter_translation)

  def parse_variant_urls(self, response) -> List[str]:
    """
    :param response: object containing html body of product page from hodinky.cz, base url
    :return: set of normalized urls to variants of given product
    """

    variants_anchor = self.watch_parameters_translation_map['variants']
    variant_urls = response.xpath(
      f"//*[contains(text(), '{variants_anchor}')]/following-sibling::*//a/@href"
    ).getall()

    if not variant_urls:
      return []

    normalized_urls = []
    for variant_url in variant_urls:
      normalized_url = response.urljoin(variant_url)
      if normalized_url != response.url:
        normalized_urls.append(normalized_url)
    return normalized_urls

  def parse_parameter(self, response, parameter_name):
    parameter_value = response.xpath(
      f"//*[contains(text(), '{parameter_name}')]/following-sibling::*/text()"
    )

    if parameter_value:
      return parameter_value.get().strip()

    return None

