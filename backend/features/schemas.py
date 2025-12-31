USER_COLS = [
    "user_id", "age", "home_country", "budget_level",
    "attr_culture", "attr_adventure", "attr_nature", 
    "attr_beaches", "attr_nightlife", "attr_cuisine", "attr_wellness",
    "ctx_urban", "ctx_seclusion"
]

DEST_COLS = [
    "dest_id", "city", "country", "region", "short_description",
    "avg_temp_monthly", "ideal_durations", "budget_level",
    "culture", "adventure", "nature", "beaches", "nightlife",
    "cuisine", "wellness", "urban", "seclusion"
]

INTERACTION_COLS = ["user_id", "dest_id", "action", "time"]

FRAUD_COLS = ["user_id", "clicks_per_min", "avg_session_time_min",
              "bookings_last_hour", "ip_entropy", "is_fraud"]

