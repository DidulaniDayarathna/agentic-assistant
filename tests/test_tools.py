from unittest.mock import MagicMock, patch

import pytest

from src.tools.calculator import CalculatorError, calculator, safe_eval
from src.tools.weather import WeatherToolError, _fetch, get_weather
from src.tools.word_count import word_count


# ---------- word_count ----------

def test_word_count_basic():
    assert word_count.invoke({"text": "hello world"}) == 2


def test_word_count_empty_string():
    assert word_count.invoke({"text": ""}) == 0


def test_word_count_extra_whitespace():
    assert word_count.invoke({"text": "  a   b  c "}) == 3


# ---------- calculator ----------

def test_calculator_basic_arithmetic():
    assert safe_eval("2 + 3 * 4") == 14


def test_calculator_functions_and_constants():
    assert safe_eval("sqrt(16)") == 4
    assert round(safe_eval("pi"), 5) == 3.14159


def test_calculator_rejects_arbitrary_code():
    with pytest.raises(CalculatorError):
        safe_eval("__import__('os').system('echo pwned')")


def test_calculator_rejects_unknown_name():
    with pytest.raises(CalculatorError):
        safe_eval("undefined_variable + 1")


def test_calculator_tool_returns_error_string_not_raise():
    result = calculator.invoke({"expression": "1 / "})
    assert result.startswith("Error:")


# ---------- weather ----------

def test_weather_fetch_success(monkeypatch):
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Kandy",
        "sys": {"country": "LK"},
        "weather": [{"description": "broken clouds"}],
        "main": {"temp": 27.68, "feels_like": 29.5, "humidity": 65},
        "wind": {"speed": 4.23},
    }
    monkeypatch.setattr("src.config.settings.openweathermap_api_key", "fake-key")
    with patch("src.tools.weather.requests.get", return_value=mock_response):
        report = _fetch("Kandy,LK")
    assert report.city == "Kandy"
    assert report.temp_c == 27.68


def test_weather_missing_api_key_returns_error_not_raise(monkeypatch):
    monkeypatch.setattr("src.config.settings.openweathermap_api_key", None)
    result = get_weather.invoke({"city": "London,GB"})
    assert "Error" in result


def test_weather_city_not_found(monkeypatch):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.ok = False
    monkeypatch.setattr("src.config.settings.openweathermap_api_key", "fake-key")
    with patch("src.tools.weather.requests.get", return_value=mock_response):
        with pytest.raises(WeatherToolError):
            _fetch("Nowhereland")
