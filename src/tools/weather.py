"""
Weather tool backed by OpenWeatherMap, with a validated structured
response instead of a free-text blob. Returning structured data makes
the tool independently testable and easier for the LLM to summarize
consistently.
"""
import logging

import requests
from langchain.tools import tool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


class WeatherReport(BaseModel):
    city: str
    country: str = ""
    condition: str
    temp_c: float
    feels_like_c: float
    humidity_pct: int
    wind_speed_ms: float

    def to_text(self) -> str:
        return (
            f"Weather in {self.city}{', ' + self.country if self.country else ''}: "
            f"{self.condition}, {self.temp_c:.1f}°C (feels like {self.feels_like_c:.1f}°C), "
            f"humidity {self.humidity_pct}%, wind {self.wind_speed_ms:.1f} m/s."
        )


class WeatherToolError(Exception):
    """Raised when the weather API can't be reached or returns bad data."""


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
def _fetch(city: str) -> WeatherReport:
    if not settings.openweathermap_api_key:
        raise WeatherToolError("OPENWEATHERMAP_API_KEY is not configured.")

    params = {
        "q": city,
        "appid": settings.openweathermap_api_key,
        "units": "metric",
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=settings.request_timeout_seconds)
    except requests.RequestException as exc:
        logger.warning("Weather API request failed: %s", exc)
        raise WeatherToolError(f"Could not reach weather service: {exc}") from exc

    if resp.status_code == 404:
        raise WeatherToolError(f"City not found: {city!r}")
    if not resp.ok:
        raise WeatherToolError(f"Weather API error ({resp.status_code}): {resp.text[:200]}")

    data = resp.json()
    try:
        return WeatherReport(
            city=data["name"],
            country=data.get("sys", {}).get("country", ""),
            condition=data["weather"][0]["description"],
            temp_c=data["main"]["temp"],
            feels_like_c=data["main"]["feels_like"],
            humidity_pct=data["main"]["humidity"],
            wind_speed_ms=data["wind"]["speed"],
        )
    except (KeyError, IndexError) as exc:
        raise WeatherToolError(f"Unexpected response shape from weather API: {exc}") from exc


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: City name, optionally with a country code, e.g. "London,GB"
              or just "Kandy".
    """
    try:
        report = _fetch(city)
        return report.to_text()
    except WeatherToolError as exc:
        # Return a clean, LLM-readable error instead of raising — a raised
        # exception here would crash the whole agent turn.
        logger.info("Weather tool returning error to agent: %s", exc)
        return f"Error fetching weather for {city!r}: {exc}"
