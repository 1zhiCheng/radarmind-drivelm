try:
    from .metrics import combine_final, combine_language, graph_question_is_eligible, match_coordinates
except ImportError:  # direct execution from this directory
    from metrics import combine_final, combine_language, graph_question_is_eligible, match_coordinates


def test_coordinate_match_uses_public_manhattan_threshold() -> None:
    result = match_coordinates(
        "<c1,CAM_FRONT,100.0,200.0> <c2,CAM_BACK,310.0,400.0>",
        "<c1,CAM_FRONT,108.0,207.0> <c2,CAM_BACK,300.0,400.0>",
    )
    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.f1 > 0.999999


def test_coordinate_match_rejects_distance_equal_to_16() -> None:
    result = match_coordinates("100.0,200.0", "108.0,208.0")
    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1


def test_graph_gating_requires_every_referenced_pair() -> None:
    graph = ((100.0, 200.0),)
    assert graph_question_is_eligible("What about 100.0,200.0?", graph)
    assert not graph_question_is_eligible("Compare 100.0,200.0 and 300.0,400.0", graph)
    assert graph_question_is_eligible("Question without an object coordinate", graph)


def test_public_score_combination() -> None:
    language = combine_language(
        {"Bleu_1": 0.3, "Bleu_2": 0.2, "Bleu_3": 0.1, "Bleu_4": 0.0,
         "ROUGE_L": 0.4, "CIDEr": 2.0}
    )
    assert abs(language - 0.25) < 1e-12
    score = combine_final(
        accuracy=0.5, planning_judge_100=80.0, language=language, match_100=60.0
    )
    assert abs(score - 0.59) < 1e-12
