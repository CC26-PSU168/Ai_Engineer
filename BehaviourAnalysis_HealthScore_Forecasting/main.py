"""
main.py — FastAPI Backend v2.0 (Step 5 + Step 7 + Step 8)

Semua endpoint:
  GET  /transactions          → ambil transaksi
  POST /transactions          → tambah transaksi
  GET  /summary               → monthly summary
  GET  /financial-score       → rule-based AI score
  GET  /behavior-analysis     → insight perilaku
  GET  /recommendations       → rekomendasi aksi
  GET  /forecast/prophet      → prediksi Prophet
  GET  /forecast/lstm         → prediksi LSTM (Step 7)
  POST /train/lstm            → training LSTM
  GET  /narrative             → narasi Gemini (Step 8)
  GET  /full-report           → semua sekaligus
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, date
import pandas as pd
import numpy as np
import sqlalchemy
import os
from dotenv import load_dotenv

load_dotenv()

from analytics_engine import generate_monthly_summary
from rule_based_ai import RuleEngine
from forecast import ExpenseForecaster
from gemininarator import get_narrator

# TensorFlow / LSTM is optional — not available on free-tier deployments
try:
    from lstm_model import load_model_and_preparer, predict_future
    LSTM_AVAILABLE = True
except (ImportError, ModuleNotFoundError, MemoryError, Exception) as e:
    LSTM_AVAILABLE = False
    load_model_and_preparer = None  # type: ignore
    predict_future = None           # type: ignore
    print(f"⚠️  TensorFlow/LSTM could not be loaded ({type(e).__name__}: {e}) — LSTM features disabled, using Prophet fallback.")

# ══════════════════════════════════════════════════════════════════
# SETUP APP
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "Personal Finance API",
    description = "Rule-Based AI + LSTM Forecasting + Gemini Narration",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

from sqlalchemy.pool import NullPool
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./finance.db")
if DATABASE_URL.startswith("sqlite"):
    engine_db = sqlalchemy.create_engine(DATABASE_URL)
else:
    engine_db = sqlalchemy.create_engine(DATABASE_URL, poolclass=NullPool)

rule_engine        = RuleEngine()
prophet_forecaster = ExpenseForecaster()
narrator           = get_narrator()

_lstm_model    = None
_lstm_preparer = None
LSTM_SAVE_DIR  = "models/saved"

if LSTM_AVAILABLE and load_model_and_preparer:
    try:
        _lstm_model, _lstm_preparer = load_model_and_preparer(LSTM_SAVE_DIR)
        print("LSTM model loaded.")
    except Exception:
        print("LSTM model belum ada. Training via POST /train/lstm.")


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def get_user_df(user_id: str) -> pd.DataFrame:
    try:
        df = pd.read_sql(
            f"SELECT * FROM transactions WHERE \"userId\" = '{user_id}'",
            con=engine_db,
        )
        if df.empty:
            # Return empty DataFrame with correct columns instead of raising 404
            return pd.DataFrame(columns=[
                "id", "userId", "type", "date", "amount",
                "category", "paymentMethod", "description",
                "notes", "isAutoCateg", "isAnomaly", "createdAt", "updatedAt"
            ])
        
        # Rename columns to match expected names
        df = df.rename(columns={
            "date": "Date",
            "amount": "Amount",
            "category": "Category",
            "type": "Transaction_Type",
            "paymentMethod": "Account_Name",
            "description": "Description",
        })
        # Convert INCOME/EXPENSE enum to title case (Income/Expense)
        df["Transaction_Type"] = df["Transaction_Type"].astype(str).str.title()
        df["Date"] = pd.to_datetime(df["Date"])
        
        return df
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Database error: {e}")



def now_my():
    n = datetime.now()
    return n.month, n.year


def ok(data: dict, period: str = None) -> dict:
    return {"status": "success", "period": period, "data": data}


# ══════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════

class TransactionIn(BaseModel):
    user_id         : str
    date            : date
    description     : str
    amount          : float = Field(..., gt=0)
    transaction_type: str   = Field(..., pattern="^(Expense|Income)$")
    category        : str
    account_name    : Optional[str] = "Default"


# ══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
def root():
    return {"app": "Personal Finance API v2.0", "docs": "/docs"}


@app.get("/transactions", tags=["Transaksi"])
def get_transactions(
    user_id : str           = Query(...),
    month   : Optional[int] = Query(None, ge=1, le=12),
    year    : Optional[int] = Query(None, ge=2000),
    category: Optional[str] = Query(None),
    limit   : int           = Query(50, ge=1, le=500),
):
    df = get_user_df(user_id)
    df["Date"] = pd.to_datetime(df["Date"])
    if month:
        df = df[df["Date"].dt.month == month]
    if year:
        df = df[df["Date"].dt.year == year]
    if category:
        df = df[df["Category"].str.lower() == category.lower()]
    df = df.sort_values("Date", ascending=False).head(limit)
    return ok({"total": len(df), "transactions": df.to_dict(orient="records")})


@app.post("/transactions", tags=["Transaksi"], status_code=201)
def add_transaction(txn: TransactionIn):
    try:
        with engine_db.connect() as conn:
            conn.execute(sqlalchemy.text("""
                INSERT INTO transactions
                    (user_id, date, description, amount,
                     transaction_type, category, account_name,
                     month, month_name, day_of_week)
                VALUES (:user_id, :date, :description, :amount,
                        :transaction_type, :category, :account_name,
                        :month, :month_name, :day_of_week)
            """), {
                "user_id"         : txn.user_id,
                "date"            : txn.date,
                "description"     : txn.description,
                "amount"          : txn.amount,
                "transaction_type": txn.transaction_type,
                "category"        : txn.category,
                "account_name"    : txn.account_name,
                "month"           : txn.date.month,
                "month_name"      : txn.date.strftime("%B"),
                "day_of_week"     : txn.date.strftime("%A"),
            })
            conn.commit()
        return {"status": "created"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/summary", tags=["Analytics"])
def get_summary(
    user_id: str = Query(...),
    month: Optional[int] = Query(None), year: Optional[int] = Query(None),
):
    m, y = month or now_my()[0], year or now_my()[1]
    df = get_user_df(user_id)
    if df.empty:
        return ok({"total_expense": 0, "total_income": 0, "top_category": "-", "period": f"{m}/{y}", "message": "Belum ada data transaksi"})
    result = generate_monthly_summary(df, month=m, year=y)
    return ok(result, period=result.get("period"))


@app.get("/financial-score", tags=["Analytics"])
def get_financial_score(
    user_id: str = Query(...),
    month: Optional[int] = Query(None), year: Optional[int] = Query(None),
):
    m, y = month or now_my()[0], year or now_my()[1]
    df = get_user_df(user_id)
    if df.empty:
        return ok({"value": 0, "status": "Belum Ada Data", "grade": "-", "deductions": [], "bonuses": [], "breakdown": {}}, period=f"{m}/{y}")
    result = rule_engine.run(df, month=m, year=y)
    return ok(result["score"], period=result["period"])


@app.get("/behavior-analysis", tags=["Analytics"])
def get_behavior_analysis(
    user_id: str = Query(...),
    month: Optional[int] = Query(None), year: Optional[int] = Query(None),
):
    m, y = month or now_my()[0], year or now_my()[1]
    df = get_user_df(user_id)
    if df.empty:
        return ok({"insights": [], "warnings": [], "positives": ["Mulai catat transaksi untuk mendapatkan insight personal."]}, period=f"{m}/{y}")
    result = rule_engine.run(df, month=m, year=y)
    return ok(result["insights"], period=result["period"])


@app.get("/recommendations", tags=["Analytics"])
def get_recommendations(
    user_id: str = Query(...),
    month: Optional[int] = Query(None), year: Optional[int] = Query(None),
):
    m, y = month or now_my()[0], year or now_my()[1]
    df = get_user_df(user_id)
    if df.empty:
        return ok({"items": [{"priority": "low", "category": "Umum", "action": "Mulai catat transaksi untuk mendapatkan rekomendasi personal dari AI.", "estimated_impact": 0}], "focus_category": "Umum", "estimated_saving": 0}, period=f"{m}/{y}")
    result = rule_engine.run(df, month=m, year=y)
    return ok(result["recommendations"], period=result["period"])


# ── Forecasting ────────────────────────────────────────────────────

@app.get("/forecast/prophet", tags=["Forecasting"])
def forecast_prophet(
    user_id : str = Query(...),
    days    : int = Query(7, ge=1, le=90),
    category: str = Query("all"),
):
    import traceback
    try:
        df = get_user_df(user_id)
        if not prophet_forecaster._trained:
            prophet_forecaster.fit(df, verbose=False)
        result = (
            prophet_forecaster.predict_all_categories(days=days)
            if category == "all"
            else prophet_forecaster.predict(days=days, category=category)
        )
        return ok(result)
    except Exception as ex:
        return {"status": "error", "message": str(ex), "traceback": traceback.format_exc()}


@app.get("/forecast/lstm", tags=["Forecasting"])
def forecast_lstm(
    user_id: str = Query(...),
    days   : int = Query(7, ge=1, le=30),
):
    """
    Prediksi LSTM. Model harus ditraining dulu via POST /train/lstm.
    MAE target ≤ 0.02 (normalized), Akurasi target ≥ 85% (MAPE < 15%).
    """
    if _lstm_model is None:
        raise HTTPException(503, "LSTM belum ditraining. POST /train/lstm dulu.")
    result = predict_future(_lstm_model, _lstm_preparer, days=days)
    return ok(result)


@app.post("/train/lstm", tags=["Training"])
def train_lstm(user_id: str = Query(...)):
    """Training LSTM dengan tf.GradientTape. Waktu: 1–5 menit."""
    global _lstm_model, _lstm_preparer

    from models.lstm_model import (
        DataPreparer, build_lstm_model,
        train_with_gradient_tape, evaluate_on_test, save_model,
    )

    df       = get_user_df(user_id)
    preparer = DataPreparer(sequence_length=14)
    data     = preparer.prepare(df, category="total")

    if data["n_train"] < 20:
        raise HTTPException(400, "Data terlalu sedikit (butuh min 50 hari).")

    model = build_lstm_model()
    train_with_gradient_tape(
        model     = model,
        X_train   = data["X_train"], y_train=data["y_train"],
        X_val     = data["X_val"],   y_val=data["y_val"],
        epochs    = 100, patience=10,
        log_dir   = f"logs/tensorboard/user_{user_id}",
    )

    eval_result = evaluate_on_test(model, preparer, data["X_test"], data["y_test"])
    save_model(model, preparer, save_dir=LSTM_SAVE_DIR)

    _lstm_model    = model
    _lstm_preparer = preparer

    return {"status": "trained", "evaluation": eval_result}


# ── Generative AI ──────────────────────────────────────────────────

@app.get("/narrative", tags=["Generative AI"])
def get_narrative(
    user_id: str           = Query(...),
    month  : Optional[int] = Query(None),
    year   : Optional[int] = Query(None),
    type   : str           = Query("summary", description="summary|score|insights|forecast"),
    days   : int           = Query(7, ge=1, le=30),
):
    """Ubah data keuangan → narasi natural language via Gemini."""
    m, y      = month or now_my()[0], year or now_my()[1]
    df        = get_user_df(user_id)
    narrative = ""
    period    = f"{m}/{y}"

    if df.empty:
        narrative = "Mulai catat transaksi kamu untuk mendapatkan analisis AI yang personal dan akurat."
        return ok({"type": type, "narrative": narrative}, period=period)

    ai_result = rule_engine.run(df, month=m, year=y)
    period    = ai_result["period"]

    if type == "summary":
        summary   = generate_monthly_summary(df, month=m, year=y)
        narrative = narrator.narrate_summary(summary, ai_result["score"])
    elif type == "score":
        s = ai_result["score"]
        narrative = narrator.narrate_score(s["value"], s["status"], s["deductions"], s["bonuses"])
    elif type == "insights":
        narrative = narrator.narrate_insights(ai_result["insights"])
    elif type == "forecast":
        try:
            if _lstm_model:
                fc = predict_future(_lstm_model, _lstm_preparer, days=days)
            else:
                # Fit prophet with user data first
                expense_df = df[df["Transaction_Type"] == "Expense"]
                if not expense_df.empty:
                    prophet_forecaster.fit(expense_df, verbose=False)
                    fc = prophet_forecaster.predict(days=days)
                    narrative = narrator.narrate_forecast(fc)
                else:
                    narrative = "Belum ada data pengeluaran untuk diprediksi."
                fc = None
            if fc:
                narrative = narrator.narrate_forecast(fc)
        except Exception as ex:
            print(f"Forecast error: {ex}")
            narrative = "Prediksi memerlukan lebih banyak data historis transaksi."
    else:
        raise HTTPException(400, "type: summary | score | insights | forecast")

    return ok({"type": type, "narrative": narrative}, period=period)


@app.get("/full-report", tags=["Analytics"])
def get_full_report(
    user_id       : str           = Query(...),
    month         : Optional[int] = Query(None),
    year          : Optional[int] = Query(None),
    forecast_days : int           = Query(7, ge=1, le=30),
    with_narrative: bool          = Query(True),
):
    """Semua data dashboard dalam satu request."""
    m, y      = month or now_my()[0], year or now_my()[1]
    df        = get_user_df(user_id)
    summary   = generate_monthly_summary(df, month=m, year=y)
    ai_result = rule_engine.run(df, month=m, year=y)

    if _lstm_model is not None:
        forecast = predict_future(_lstm_model, _lstm_preparer, days=forecast_days)
        fc_model = "lstm"
    else:
        if not prophet_forecaster._trained:
            prophet_forecaster.fit(df, verbose=False)
        forecast = prophet_forecaster.predict(days=forecast_days)
        fc_model = "prophet"

    result = {
        "summary"        : summary,
        "score"          : ai_result["score"],
        "insights"       : ai_result["insights"],
        "recommendations": ai_result["recommendations"],
        "forecast"       : {**forecast, "model_used": fc_model},
    }

    if with_narrative:
        result["narrative"] = {
            "summary" : narrator.narrate_summary(summary, ai_result["score"]),
            "score"   : narrator.narrate_score(
                            ai_result["score"]["value"],
                            ai_result["score"]["status"],
                            ai_result["score"]["deductions"],
                        ),
            "forecast": narrator.narrate_forecast(forecast),
        }

    return ok(result, period=ai_result["period"])