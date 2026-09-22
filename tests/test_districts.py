from datetime import date

from utils.districts import locate_district, season_for_date, plausible_crops

NAGPUR_POINT = {
    "type": "Polygon",
    "coordinates": [[
        [79.0, 21.0], [79.05, 21.0], [79.05, 21.05], [79.0, 21.05], [79.0, 21.0]
    ]],
}


def test_locate_district_within_bbox():
    assert locate_district(NAGPUR_POINT) == "Nagpur"


def test_season_for_date():
    assert season_for_date(date(2026, 7, 15)) == "kharif"
    assert season_for_date(date(2026, 1, 15)) == "rabi"


def test_plausible_crops_reflect_district_emphasis():
    crops = plausible_crops("Nagpur", "kharif")
    assert "cotton" in crops
    assert crops[0] == "cotton"
