"""
Car Price Predictor - simple web interface for the notebook's model.

Pipeline (same as car_price_prediction.ipynb):
  - drop duplicates
  - brand = first word of "name"
  - OneHotEncoder on brand, fuel, seller_type, transmission, owner
  - year + km_driven passed through
  - LinearRegression, 80/20 split, random_state=42

Run:  python app.py   (it prints the address and opens your browser)
Needs car_data.csv in the same folder (or set CSV_PATH env var).
"""
import os
import sys

import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

CSV_PATH = os.environ.get("CSV_PATH", "car_data.csv")
# Prices in the dataset are in Indian Rupees (INR).
# Approximate exchange rate used for the Tunisian Dinar line - edit it if you like.
INR_TO_TND = 0.031
WORDS_COLUMNS = ["brand", "fuel", "seller_type", "transmission", "owner"]
NUM_COLUMNS = ["year", "km_driven"]

# ---------------------------------------------------------------- training
if not os.path.exists(CSV_PATH):
    sys.exit(f"ERROR: '{CSV_PATH}' not found. Put car_data.csv next to app.py.")

df = pd.read_csv(CSV_PATH).drop_duplicates().copy()
df["brand"] = df["name"].str.split().str[0]

X = df.drop(columns=["selling_price", "name"])[WORDS_COLUMNS + NUM_COLUMNS]
y = df["selling_price"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("with words", OneHotEncoder(handle_unknown="ignore"), WORDS_COLUMNS),
        ("numcolums", "passthrough", NUM_COLUMNS),
    ]
)
model = LinearRegression()
model.fit(preprocessor.fit_transform(x_train), y_train)

pred_test = model.predict(preprocessor.transform(x_test))
METRICS = {
    "r2": round(float(r2_score(y_test, pred_test)), 3),
    "mae": round(float(mean_absolute_error(y_test, pred_test))),
    "rows": int(len(df)),
}

OPTIONS = {col: sorted(df[col].dropna().unique().tolist()) for col in WORDS_COLUMNS}
YEAR_RANGE = [int(df["year"].min()), int(df["year"].max())]

# ---------------------------------------------------------------- web app
app = Flask(__name__)

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Car Price Predictor</title>
<style>
  :root { --bg:#f4f6fb; --card:#fff; --text:#1c2333; --muted:#6b7385;
          --accent:#2f5bea; --border:#dfe3ee; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#12151d; --card:#1b1f2b; --text:#e8ebf3; --muted:#9097a8;
            --accent:#6b8cff; --border:#2c3244; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, sans-serif;
         background:var(--bg); color:var(--text); display:flex;
         justify-content:center; padding:32px 16px; }
  .card { background:var(--card); border:1px solid var(--border);
          border-radius:16px; padding:28px; width:100%; max-width:520px; }
  h1 { margin:0 0 4px; font-size:1.5rem; }
  .sub { color:var(--muted); font-size:.9rem; margin-bottom:22px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  .full { grid-column:1 / -1; }
  label { display:block; font-size:.8rem; color:var(--muted); margin-bottom:5px; }
  select, input { width:100%; padding:10px 12px; font-size:.95rem;
          border:1px solid var(--border); border-radius:10px;
          background:var(--bg); color:var(--text); }
  button { margin-top:20px; width:100%; padding:13px; font-size:1rem;
           font-weight:600; color:#fff; background:var(--accent);
           border:0; border-radius:10px; cursor:pointer; }
  button:disabled { opacity:.6; cursor:wait; }
  #result { margin-top:20px; padding:18px; border-radius:12px; text-align:center;
            background:var(--bg); border:1px dashed var(--border); display:none; }
  #result .label { color:var(--muted); font-size:.8rem; }
  #result .price { font-size:2rem; font-weight:700; color:var(--accent); }
  #result .alt { margin-top:6px; color:var(--muted); font-size:.95rem; }
  #result.error .price { color:#d64545; font-size:1rem; }
  .foot { margin-top:18px; font-size:.75rem; color:var(--muted); text-align:center; }
  @media (max-width:480px) { .grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="card">
  <h1>🚗 Car Price Predictor</h1>
  <div class="sub">Enter the car's details to estimate its selling price.</div>

  <div class="grid">
    <div class="full"><label>Brand</label><select id="brand"></select></div>
    <div><label>Year</label>
      <input id="year" type="number" min="{{ yr[0] }}" max="{{ yr[1] + 1 }}" value="{{ yr[1] - 5 }}"></div>
    <div><label>Kilometers driven</label>
      <input id="km_driven" type="number" min="0" value="50000"></div>
    <div><label>Fuel</label><select id="fuel"></select></div>
    <div><label>Transmission</label><select id="transmission"></select></div>
    <div><label>Seller type</label><select id="seller_type"></select></div>
    <div><label>Owner</label><select id="owner"></select></div>
  </div>

  <button id="go">Predict price</button>

  <div id="result">
    <div class="label">Estimated selling price</div>
    <div class="price" id="price"></div>
    <div class="alt" id="alt"></div>
  </div>

  <div class="foot">Linear Regression · trained on {{ m.rows }} cars ·
    R² = {{ m.r2 }} · avg. error ≈ ₹{{ "{:,}".format(m.mae) }}<br>Prices are in Indian Rupees (₹)</div>
</div>

<script>
const OPTIONS = {{ options | tojson }};
for (const [id, values] of Object.entries(OPTIONS)) {
  const el = document.getElementById(id);
  el.innerHTML = values.map(v => `<option value="${v}">${v}</option>`).join("");
}
const FIELDS = ["brand","fuel","seller_type","transmission","owner","year","km_driven"];
const btn = document.getElementById("go");
const box = document.getElementById("result");
const priceEl = document.getElementById("price");
const altEl = document.getElementById("alt");

btn.addEventListener("click", async () => {
  const payload = {};
  FIELDS.forEach(f => payload[f] = document.getElementById(f).value);
  btn.disabled = true; btn.textContent = "Predicting…";
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    box.style.display = "block";
    if (res.ok) {
      box.classList.remove("error");
      priceEl.textContent = "₹ " + data.formatted;
      altEl.textContent = "≈ " + data.tnd + " TND (approx.)";
    } else {
      box.classList.add("error");
      priceEl.textContent = data.error || "Something went wrong";
      altEl.textContent = "";
    }
  } catch (e) {
    box.style.display = "block"; box.classList.add("error");
    priceEl.textContent = "Could not reach the server"; altEl.textContent = "";
  }
  btn.disabled = false; btn.textContent = "Predict price";
});
</script>
</body>
</html>"""


@app.get("/")
def index():
    return render_template_string(PAGE, options=OPTIONS, m=METRICS, yr=YEAR_RANGE)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        row = {c: data[c] for c in WORDS_COLUMNS}
        row["year"] = int(data["year"])
        row["km_driven"] = float(data["km_driven"])
    except (KeyError, ValueError, TypeError):
        return jsonify(error="Please fill in every field with valid values."), 400

    sample = pd.DataFrame([row])[WORDS_COLUMNS + NUM_COLUMNS]
    price = float(model.predict(preprocessor.transform(sample))[0])
    price = max(price, 0.0)  # linear regression can go negative on extreme inputs
    return jsonify(
        price=round(price),
        formatted=f"{price:,.0f}",
        tnd=f"{price * INR_TO_TND:,.0f}",
    )


if __name__ == "__main__":
    import socket
    import threading
    import webbrowser

    def free_port(start=8000):
        for p in range(start, start + 50):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("127.0.0.1", p)) != 0:
                    return p
        return start

    port = int(os.environ.get("PORT", free_port()))
    url = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 50)
    print(f"  Model trained (R2 = {METRICS['r2']}).")
    print(f"  Open this address in your browser:  {url}")
    print("  Keep this window open. Press Ctrl+C to stop.")
    print("=" * 50 + "\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False)
