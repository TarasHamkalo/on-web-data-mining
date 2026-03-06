from scrapy.item import Item, Field

# class WatchVariantItem(Item):
#     url = Field()
#     image_url = Field()
#
#     case_material = Field()
#     strap_material = Field()
#     strap_color = Field()
#     dial_color = Field()
#
# class ShopAlternativeItem(Item):
#     url = Field()
#     price = Field()
#     currency = Field()

class WatchItem(Item):
    url = Field()
    title = Field()
    brand = Field()

    price = Field()
    currency = Field()
    image_url = Field()

    case_material = Field()
    strap_material = Field()
    strap_color = Field()
    dial_color = Field()

    functions = Field()
    # variants = Field(serializer=WatchVariantItem)
    # shop_alternatives = Field(serializer=ShopAlternativeItem)