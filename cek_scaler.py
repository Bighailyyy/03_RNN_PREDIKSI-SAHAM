import joblib

for nama in ["BBCA", "BBRI"]:
    s = joblib.load(f"hasil-{nama}/scaler.pkl")
    print(nama, "-> min:", s.data_min_, "| max:", s.data_max_)