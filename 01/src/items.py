from scrapy.item import Item, Field

class BaseWatchItem(Item):
    url = Field()

    price = Field()
    currency = Field()

    image_url = Field()

    case_material = Field()
    strap_material = Field()
    strap_color = Field()
    dial_color = Field()


# class ShopAlternativeItem(Item):
#     url = Field()
#     price = Field()
#     currency = Field()

class WatchItem(BaseWatchItem):
    title = Field()
    brand = Field()

    functions = Field()
    variants = Field()

    _variant_urls = Field()