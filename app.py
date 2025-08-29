# app.py

from flask import Flask, request, jsonify
import requests
from datetime import datetime
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__)


# Load Gemini API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API not found in .env")
genai.configure(api_key=GEMINI_API_KEY)
# Load tariffs from JSON file at startup
TARIFFS_PATH = "tariffs.json"
if not os.path.exists(TARIFFS_PATH):
    raise FileNotFoundError(f"{TARIFFS_PATH} not found. Please create it with tariff data.")

with open(TARIFFS_PATH, "r") as f:
    TARIFFS = json.load(f)




# Gemini 2.5 Pro call
def call_gemini(system_prompt, user_prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = system_prompt + "\n" + user_prompt
    response = model.generate_content(prompt)
    return response.text

def parse_gemini_json(gemini_response):
    try:
        # Gemini may return code block, plain text, or explanation
        text = gemini_response.strip()
        # Remove code block markers and any leading/trailing text
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        # Remove any leading text before the first '{'
        json_start = text.find('{')
        if json_start != -1:
            text = text[json_start:]
        # Remove any trailing text after the last '}'
        json_end = text.rfind('}')
        if json_end != -1:
            text = text[:json_end+1]
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Failed to parse Gemini JSON: {e}")
def get_tariff(state, area, vehicle_type):
    try:
        return TARIFFS[state][area][vehicle_type]
    except KeyError:
        return None

def compute_price(start_dt, end_dt, tariff):
    duration = end_dt - start_dt
    total_minutes = duration.total_seconds() / 60
    if total_minutes <= tariff.get("grace_minutes", 0):
        return 0

    hours = total_minutes / 60
    if tariff.get("rounding_rule") == "ceil_to_next_hour":
        billable_hours = int(hours) if hours == int(hours) else int(hours) + 1
    else:
        billable_hours = hours

    weekend = start_dt.weekday() >= 5 or end_dt.weekday() >= 5
    multiplier = tariff.get("weekend_multiplier", 1.0) if weekend else 1.0

    price = billable_hours * tariff["hourly_rate"] * multiplier
    price = min(price, tariff["daily_cap"])
    return round(price, 2)

@app.route('/predict_parking_price', methods=['POST'])
def predict_parking_price():
    data = request.json
    user_prompt = (
        f"Start date: {data['start_date']}, start time: {data['start_time']}, "
        f"end date: {data['end_date']}, end time: {data['end_time']}, "
        f"area: {data['area']}, state: {data['state']}, vehicle: {data['vehicle_type']}."
        " Extract and validate these inputs for a parking tariff calculator. "
        "Return ONLY JSON with: normalized ISO-8601 datetimes, area, state, vehicle_type, "
        "and any validation errors. Do not compute prices."
    )
    system_prompt = (
        "You extract and validate inputs for a parking tariff calculator. "
        "Respond ONLY with valid JSON, starting with '{' and matching this schema: "
        "{'inputs': {'start_datetime': str, 'end_datetime': str, 'area': str, 'state': str, 'vehicle_type': str}, "
        "'validation': {'is_valid': bool, 'errors': list}}. "
        "Do not include any text, explanation, or code block formatting. Do not compute prices."
    )

    try:
        gemini_response = call_gemini(system_prompt, user_prompt)
        parsed = parse_gemini_json(gemini_response)
        if not parsed["validation"]["is_valid"]:
            return jsonify({"error": parsed["validation"]["errors"]}), 400

        # Parse datetimes
        start_dt = datetime.fromisoformat(parsed["inputs"]["start_datetime"])
        end_dt = datetime.fromisoformat(parsed["inputs"]["end_datetime"])
        state = parsed["inputs"]["state"]
        area = parsed["inputs"]["area"]
        vehicle_type = parsed["inputs"]["vehicle_type"]

        tariff = get_tariff(state, area, vehicle_type)
        if not tariff:
            return jsonify({"error": f"No tariff found for {state} / {area} / {vehicle_type}"}), 400

        price = compute_price(start_dt, end_dt, tariff)
        return jsonify({
            "inputs": parsed["inputs"],
            "tariff": tariff,
            "price": price
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
