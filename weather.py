import requests
from datetime import datetime, timedelta


class UnifiedWeatherAgent:
    def __init__(self, app_name="Verda", email="mlayeladam@gmail.com"):
        self.metno_url = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
        self.metno_headers = {"User-Agent": f"{app_name} {email}"}

    def get_yesterday_weather(self, latitude, longitude):
        yesterday = datetime.utcnow() - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")

        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={latitude}&longitude={longitude}"
            f"&start_date={date_str}&end_date={date_str}"
            f"&hourly=temperature_2m,relative_humidity_2m,precipitation"
            f"&timezone=auto"
        )

        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        hourly = data.get("hourly", {})

        temps = [t for t in hourly.get("temperature_2m", []) if t is not None]
        humidity = [h for h in hourly.get("relative_humidity_2m", []) if h is not None]
        rain = [r for r in hourly.get("precipitation", []) if r is not None]

        return {
            "avg_temp": round(sum(temps) / len(temps), 1) if temps else None,
            "avg_humidity": round(sum(humidity) / len(humidity), 1) if humidity else None,
            "total_precipitation": round(sum(rain), 2) if rain else None
        }

    def get_today_forecast(self, latitude, longitude, hours=6):
        params = {"lat": latitude, "lon": longitude}
        response = requests.get(self.metno_url, headers=self.metno_headers, params=params)
        response.raise_for_status()
        data = response.json()

        timeseries = data["properties"]["timeseries"][:hours]
        forecast = []
        for entry in timeseries:
            time = entry["time"]
            details = entry["data"]["instant"]["details"]
            next_hour = entry["data"].get("next_1_hours", {})

            forecast.append({
                "time": time,
                "temp": details.get("air_temperature"),
                "humidity": details.get("relative_humidity"),
                "precip": next_hour.get("precipitation_amount", 0),
                "symbol": next_hour.get("summary", {}).get("symbol_code", "n/a")
            })
        return forecast

    def is_weather_suitable(self, forecast):
        temps = [f["temp"] for f in forecast if f["temp"] is not None]
        rain = sum(f["precip"] for f in forecast if f["precip"] is not None)
        avg_temp = sum(temps) / len(temps) if temps else 0

        return {
            "suitable_for_planting": 18 <= avg_temp <= 30 and rain < 5,
            "avg_forecast_temp": round(avg_temp, 1),
            "total_forecast_rain": round(rain, 2)
        }

    def get_normalized_weather_report(self, latitude, longitude, hours=6):
        yesterday = self.get_yesterday_weather(latitude, longitude)
        today = self.get_today_forecast(latitude, longitude, hours)
        suitability = self.is_weather_suitable(today)

        return {
            "yesterday_weather": yesterday,
            "today_forecast": today,
            "planting_suitability": suitability
        }


# Example usage (for testing)
if __name__ == "__main__":
    agent = UnifiedWeatherAgent()
    lat = 35.78  # Example: Monastir, Tunisia
    lon = 10.82

    report = agent.get_normalized_weather_report(lat, lon)
    from pprint import pprint
    pprint(report)
