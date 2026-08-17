from __future__ import annotations

import json

from infer_graph_rollout import rollout_messages


def sample_record() -> dict:
    return {
        "id": "scene_frame",
        "system": "graph system",
        "images": {
            camera: f"/{camera}.jpg"
            for camera in (
                "CAM_FRONT",
                "CAM_FRONT_LEFT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            )
        },
        "nodes": [
            {
                "id": "p0",
                "task": "perception",
                "question": "perception question",
                "answer": "GOLD_PERCEPTION_SECRET",
            },
            {
                "id": "p1",
                "task": "prediction",
                "question": "prediction question",
                "answer": "GOLD_PREDICTION_SECRET",
            },
        ],
    }


def test_rollout_uses_predicted_upstream_context_only() -> None:
    rendered = json.dumps(
        rollout_messages(sample_record(), {"p0": "MODEL_PERCEPTION_PREDICTION"}, 1)
    )
    assert "MODEL_PERCEPTION_PREDICTION" in rendered
    assert "GOLD_PERCEPTION_SECRET" not in rendered
    assert "GOLD_PREDICTION_SECRET" not in rendered
    assert "prediction question" in rendered


def test_first_node_contains_six_images_once() -> None:
    messages = rollout_messages(sample_record(), {}, 0)
    image_items = [
        item
        for message in messages
        for item in message["content"]
        if item.get("type") == "image"
    ]
    assert len(image_items) == 6
