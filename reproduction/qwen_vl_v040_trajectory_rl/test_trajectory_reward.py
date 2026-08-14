from trajectory_reward import compute_score


SOURCE = "radarmind_drivelm_trajectory_planning"


def score(prediction, reference, allowed=None):
    return compute_score(
        SOURCE, prediction, reference,
        {"allowed_object_ids": allowed or []},
    )


def test_exact_reference_has_unit_reward():
    answer = "The ego vehicle should slow down and yield to <c1,CAM_FRONT,10.0,20.0>."
    assert score(answer, answer, ["<c1,CAM_FRONT,10.0,20.0>"])["score"] == 1.0


def test_action_mismatch_is_penalized():
    good = score("The ego vehicle should brake.", "The ego vehicle should brake.")["score"]
    bad = score("The ego vehicle should accelerate.", "The ego vehicle should brake.")["score"]
    assert good > bad + 0.4


def test_hallucinated_object_is_penalized():
    ref = "The ego vehicle should slow down."
    safe = score("The ego vehicle should slow down.", ref)["score"]
    hallucinated = score("The ego vehicle should slow down for <c9,CAM_BACK,1.0,2.0>.", ref)["score"]
    assert safe > hallucinated


def test_empty_response_is_low_reward():
    assert score("", "The ego vehicle should brake.")["score"] < 0.1
