def test_imports():
    from ssfaitk.models import EffortClassifier, GearPredictor, VesselTypePredictor

    assert EffortClassifier is not None
    assert GearPredictor is not None
    assert VesselTypePredictor is not None
