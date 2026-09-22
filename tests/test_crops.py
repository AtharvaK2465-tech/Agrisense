from utils.crops import classify_ndvi_series, CROP_SIGNATURES


def test_classifier_recovers_exact_reference_signatures():
    for crop, signature in CROP_SIGNATURES.items():
        best, scores = classify_ndvi_series(signature)
        assert best == crop
        assert scores[best] == max(scores.values())


def test_scores_sum_to_one():
    _, scores = classify_ndvi_series(CROP_SIGNATURES["cotton"])
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_noisy_signature_still_classifies_correctly():
    noisy = [v + 0.01 for v in CROP_SIGNATURES["sugarcane"]]
    best, _ = classify_ndvi_series(noisy)
    assert best == "sugarcane"
