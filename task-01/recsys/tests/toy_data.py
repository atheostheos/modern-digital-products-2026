"""Tiny raw Amazon-like dataset shared by integration and e2e tests."""

from datetime import UTC, datetime

TOY_CATEGORY = "Toy_Category"


def ms(*ymd: int) -> int:
    return int(datetime(*ymd, tzinfo=UTC).timestamp() * 1000)


REVIEWS = [
    ("u1", "p1", ms(2022, 1, 5)),
    ("u1", "p2", ms(2022, 11, 1)),  # val
    ("u1", "p3", ms(2023, 2, 1)),  # test, warm user
    ("u2", "p1", ms(2022, 3, 1)),
    ("u2", "p1", ms(2022, 4, 1)),  # duplicate pair -> deduplicated
    ("u3", "p2", ms(2023, 1, 15)),  # test, cold user
    ("u4", "p1", ms(2023, 6, 1)),  # after test_end -> dropped
]
META = [
    {
        "parent_asin": "p1",
        "title": "Guitar",
        "store": "Fender",
        "categories": ["Musical Instruments", "Guitars", "Electric Guitars"],
        "price": 500.0,
    },
    {
        "parent_asin": "p2",
        "title": "Strings",
        "store": "Ernie Ball",
        "categories": ["Musical Instruments", "Instrument Accessories", "Strings"],
        "price": "5",
    },
    {"parent_asin": "p3", "title": "Picks", "store": None, "categories": [], "price": None},
]
