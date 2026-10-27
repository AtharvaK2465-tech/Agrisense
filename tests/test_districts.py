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
    assert "no_crop" in crops


def test_detect_ground_context_vit_campus():
    from utils.districts import detect_ground_context
    vit_ground = {
        "type": "Polygon",
        "coordinates": [[
            [73.868, 18.463], [73.870, 18.463], [73.870, 18.465], [73.868, 18.465], [73.868, 18.463]
        ]],
    }
    res = detect_ground_context(vit_ground, mean_ndvi=0.09)
    assert res["is_non_agri"] is True
    assert "VIT" in res["category"]


def test_low_ndvi_agricultural_field_is_not_no_crop():
    """A stressed cotton field in Hingoli (low NDVI ~0.15) must NOT be tagged as non-agricultural.
    Only geographic/OSM detection should trigger is_non_agri, not NDVI alone."""
    from utils.districts import detect_ground_context
    hingoli_field = {
        "type": "Polygon",
        "coordinates": [[
            [77.1, 19.7], [77.15, 19.7], [77.15, 19.75], [77.1, 19.75], [77.1, 19.7]
        ]],
    }
    res = detect_ground_context(hingoli_field, mean_ndvi=0.15)
    assert res["is_non_agri"] is False

