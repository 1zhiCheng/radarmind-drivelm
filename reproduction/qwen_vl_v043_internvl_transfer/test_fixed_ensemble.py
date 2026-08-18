from build_fixed_ensemble import use_grounding


def record(index: int, tag: list[int]) -> dict:
    return {"qa_index": index, "tag": tag}


def test_anchor_mode_routes_only_frame_anchor() -> None:
    assert use_grounding(record(0, [2]), "anchor")
    assert not use_grounding(record(1, [3]), "anchor")


def test_anchor_tag3_routes_anchor_and_tag3() -> None:
    assert use_grounding(record(0, [2]), "anchor_tag3")
    assert use_grounding(record(2, [3]), "anchor_tag3")
    assert not use_grounding(record(3, [1]), "anchor_tag3")
