from kmu26_auv_vla_data_collector.contract import ACTION_NAMES, STATE_NAMES
from kmu26_auv_vla_data_collector.export_lerobot import VIDEO_KEYS, _features, _modality


def test_exported_modality_matches_model_contract():
    modality = _modality()

    assert modality["state"]["validity"]["end"] == len(STATE_NAMES)
    assert modality["action"]["motion"]["end"] == len(ACTION_NAMES)
    assert modality["video"]["ego"]["original_key"] == "observation.images.ego"
    assert (
        modality["video"]["buoy_release"]["original_key"]
        == "observation.images.buoy_release"
    )


def test_v3_feature_schema_uses_named_axes():
    features = _features({VIDEO_KEYS[0]: (480, 640), VIDEO_KEYS[1]: (360, 640)})

    assert features["observation.state"]["shape"] == (len(STATE_NAMES),)
    assert features["observation.state"]["names"]["axes"] == list(STATE_NAMES)
    assert features["action"]["names"]["axes"] == list(ACTION_NAMES)
    assert features[VIDEO_KEYS[0]]["dtype"] == "video"
    assert features[VIDEO_KEYS[0]]["shape"] == (480, 640, 3)
