from utils import ai_engine


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = ai_engine.generate_ai_explanation(
        top_vendor={
            "vendor_name": "Test Vendor",
            "overall_score": 90,
            "rank": 1,
            "cost_score": 80,
            "quality_score": 95,
            "delivery_score": 90,
            "reliability_score": 92,
            "sustainability_score": 85,
            "cost_contribution": 24,
            "quality_contribution": 28.5,
            "delivery_contribution": 18,
            "reliability_contribution": 13.8,
            "sustainability_contribution": 4.25,
        },
        runner_up=None,
        weights={
            "Cost": 30,
            "Quality": 30,
            "Delivery": 20,
            "Reliability": 15,
            "Sustainability": 5,
        },
    )
    assert result["success"] is False
    assert "API key" in result["message"]
