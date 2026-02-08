from main1 import get_weather

def test_get_weather():
    assert get_weather(21) == "hot"