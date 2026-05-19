"""


Fungsi utama:
  GeminiNarrator.narrate()  → ubah dict insight → kalimat natural language
  GeminiNarrator.summarize() → rangkum laporan keuangan bulanan

"""

import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Load API key dari environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════
# TEMPLATE PROMPT
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Kamu adalah asisten keuangan personal yang ramah dan membantu.
Tugasmu mengubah data keuangan menjadi narasi yang mudah dipahami
oleh pengguna awam.

Aturan:
- Gunakan Bahasa Indonesia yang santai tapi profesional
- Jangan sebut angka mentah terlalu banyak, fokus pada insight
- Maksimal 3-4 kalimat per narasi
- Nada positif dan konstruktif, bukan menghakimi
- Jika ada hal positif, apresiasi dulu sebelum saran
"""

NARRATE_PROMPT = """
Berikut data keuangan bulan ini:
{data}

Ubah data di atas menjadi narasi singkat yang informatif dan mudah dipahami.
Jangan ulangi semua angka, fokus pada insight utama.
"""

SUMMARY_PROMPT = """
Berikut ringkasan keuangan lengkap:
{data}

Buatkan ringkasan eksekutif dalam 3-4 kalimat yang menjelaskan:
1. Kondisi keuangan secara keseluruhan
2. Hal paling penting yang perlu diperhatikan
3. Satu rekomendasi utama

Gunakan bahasa yang mudah dipahami.
"""

SCORE_PROMPT = """
Skor keuangan pengguna: {score}/100 dengan status "{status}".

Detail:
{data}

Jelaskan kondisi keuangan ini dalam 2-3 kalimat. 
Sebutkan apa yang sudah baik dan apa yang perlu diperbaiki.
"""

FORECAST_PROMPT = """
Prediksi pengeluaran {days} hari ke depan:
{data}

Sampaikan prediksi ini kepada pengguna dalam 2-3 kalimat.
Sebutkan total prediksi, hari tersibuk, dan satu tips relevan.
"""


# ══════════════════════════════════════════════════════════════════
# FALLBACK NARRATOR — tanpa API key
# ══════════════════════════════════════════════════════════════════

class FallbackNarrator:
    """
    Narrator berbasis template — dipakai saat Gemini tidak tersedia.
    Hasilnya tidak senatural Gemini tapi tetap informatif.
    """

    def narrate_insights(self, insights: dict) -> str:
        parts = []

        warnings  = insights.get("warnings", [])
        positives = insights.get("positives", [])
        insight_l = insights.get("insights", [])

        if positives:
            parts.append(positives[0])
        if warnings:
            parts.append(warnings[0])
        if insight_l and not parts:
            parts.append(insight_l[0])

        return " ".join(parts) if parts else "Tidak ada insight tersedia."

    def narrate_score(self, score: int, status: str, deductions: list) -> str:
        base = f"Skor keuangan kamu bulan ini adalah {score}/100 — {status}."
        if deductions:
            top = deductions[0]
            base += f" Hal utama yang memengaruhi: {top['rule']}."
        return base

    def narrate_forecast(self, forecast: dict) -> str:
        # Handle both nested (from _predict_total) and flat (predict_all_categories) structure
        summary = forecast.get("summary", forecast)
        total  = summary.get("total_predicted", forecast.get("total_predicted", 0))
        peak   = summary.get("peak_day_name",   forecast.get("peak_day_name", "-"))
        days   = forecast.get("days_ahead", 7)
        avg    = summary.get("avg_per_day",     forecast.get("avg_per_day", 0))
        period = forecast.get("period", f"{days} hari ke depan")
        return (
            f"Prediksi pengeluaran {days} hari ke depan ({period}) "
            f"sekitar Rp {total:,.0f} atau rata-rata Rp {avg:,.0f} per hari. "
            f"Hari dengan pengeluaran tertinggi diprediksi pada {peak}."
        )

    def narrate_summary(self, summary: dict, score: dict) -> str:
        expense = summary.get("total_expense", 0)
        income  = summary.get("total_income", 0)
        top_cat = summary.get("top_category", "-")
        sc      = score.get("value", 0)
        status  = score.get("status", "-")
        return (
            f"Bulan {summary.get('period', 'ini')}, total pengeluaran "
            f"Rp {expense:,.0f} dari pemasukan Rp {income:,.0f}. "
            f"Kategori terbesar adalah {top_cat}. "
            f"Skor keuangan kamu {sc}/100 — {status}."
        )


# ══════════════════════════════════════════════════════════════════
# GEMINI NARRATOR — dengan API key
# ══════════════════════════════════════════════════════════════════

class GeminiNarrator:
    """
    Ubah data keuangan menjadi narasi natural language menggunakan Gemini.

    Cara pakai:
        narrator = GeminiNarrator(api_key="your_key")
        text = narrator.narrate_insights(insights_dict)
        text = narrator.narrate_score(74, "Cukup Stabil", deductions)
        text = narrator.narrate_forecast(forecast_dict)
        text = narrator.narrate_summary(summary_dict, score_dict)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key  = api_key or GEMINI_API_KEY
        self._ready   = False
        self._model   = None
        self._fallback = FallbackNarrator()

        if not self.api_key or self.api_key == "":
            print("WARNING: Modul google.generativeai tidak ditemukan. Menggunakan fallback narrator.")
            print("   Isi di .env: GEMINI_API_KEY=your_key_here")
            print("   Atau: narrator = GeminiNarrator(api_key='your_key')")
            return

        if not GEMINI_AVAILABLE:
            print("ERROR: google-generativeai belum terinstall.")
            print("   Jalankan: pip install google-generativeai")
            return

        try:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(
                model_name    = "gemini-1.5-flash",
                system_instruction = SYSTEM_PROMPT,
            )
            self._ready = True
            print("SUCCESS: Gemini API berhasil diinisialisasi.")
        except Exception as e:
            print(f"ERROR: Gagal inisialisasi Gemini: {e}")

    def _call(self, prompt: str) -> str:
        """Panggil Gemini API dengan error handling."""
        if not self._ready:
            return None

        try:
            response = self._model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}. Menggunakan fallback.")
            return None

    # ── NARRATE INSIGHTS ─────────────────────────────────────────

    def narrate_insights(self, insights: dict) -> str:
        """
        Ubah dict insights (warnings, positives, insights) → narasi.

        Input contoh:
          {
            "warnings" : ["Pengeluaran hiburan naik 25%"],
            "positives": ["Savings rate 22% — di atas target"],
            "insights" : ["Total 87 transaksi bulan ini"]
          }
        """
        prompt   = NARRATE_PROMPT.format(data=json.dumps(insights, ensure_ascii=False, indent=2))
        result   = self._call(prompt)
        return result if result else self._fallback.narrate_insights(insights)

    # ── NARRATE SCORE ─────────────────────────────────────────────

    def narrate_score(self, score: int, status: str, deductions: list, bonuses: list = None) -> str:
        """
        Ubah skor keuangan → narasi penjelasan.

        Input contoh:
          score=74, status="Cukup Stabil",
          deductions=[{"rule": "Tabungan di bawah ideal", ...}]
        """
        data = {
            "score"     : score,
            "status"    : status,
            "deductions": deductions[:3],   # ambil 3 teratas
            "bonuses"   : bonuses or [],
        }
        prompt = SCORE_PROMPT.format(
            score  = score,
            status = status,
            data   = json.dumps(data, ensure_ascii=False, indent=2),
        )
        result = self._call(prompt)
        return result if result else self._fallback.narrate_score(score, status, deductions)

    # ── NARRATE FORECAST ──────────────────────────────────────────

    def narrate_forecast(self, forecast: dict) -> str:
        """
        Ubah hasil prediksi LSTM/Prophet → narasi.

        Input contoh:
          {
            "days_ahead": 7,
            "total_predicted": 1500000,
            "peak_day_name": "Saturday",
            "daily": [...]
          }
        """
        # Ringkas sebelum dikirim (hemat token)
        summary_data = {
            "days_ahead"     : forecast.get("days_ahead"),
            "total_predicted": forecast.get("total_predicted"),
            "avg_per_day"    : forecast.get("avg_per_day"),
            "peak_day"       : forecast.get("peak_day_name"),
            "peak_amount"    : forecast.get("peak_amount"),
        }
        prompt = FORECAST_PROMPT.format(
            days = forecast.get("days_ahead", 7),
            data = json.dumps(summary_data, ensure_ascii=False, indent=2),
        )
        result = self._call(prompt)
        return result if result else self._fallback.narrate_forecast(forecast)

    # ── NARRATE FULL SUMMARY ──────────────────────────────────────

    def narrate_summary(self, summary: dict, score: dict) -> str:
        """
        Buat ringkasan eksekutif dari summary + score.
        Ini yang ditampilkan di halaman utama dashboard.
        """
        combined = {
            "period"        : summary.get("period"),
            "total_expense" : summary.get("total_expense"),
            "total_income"  : summary.get("total_income"),
            "net_cashflow"  : summary.get("net_cashflow"),
            "top_category"  : summary.get("top_category"),
            "score"         : score.get("value"),
            "status"        : score.get("status"),
            "grade"         : score.get("grade"),
        }
        prompt = SUMMARY_PROMPT.format(
            data=json.dumps(combined, ensure_ascii=False, indent=2)
        )
        result = self._call(prompt)
        return result if result else self._fallback.narrate_summary(summary, score)


# ══════════════════════════════════════════════════════════════════
# HELPER — buat GeminiNarrator dari .env otomatis
# ══════════════════════════════════════════════════════════════════

def get_narrator(api_key: Optional[str] = None) -> GeminiNarrator:
    return GeminiNarrator(api_key=api_key)


if __name__ == "__main__":
    # Isi API key di sini untuk test langsung
    API_KEY = "AIzaSyCh5QU-aPrtRrMRsshfA4B5JuoUAXKB2V8"

    narrator = GeminiNarrator(api_key=API_KEY)

    print("\n=== TEST: narrate_insights ===")
    print(narrator.narrate_insights({
        "warnings" : ["Pengeluaran hiburan naik 25% dari bulan lalu"],
        "positives": ["Savings rate 22% — sudah di atas target 20%"],
        "insights" : ["Total 87 transaksi pengeluaran bulan ini"],
    }))

    print("\n=== TEST: narrate_score ===")
    print(narrator.narrate_score(
        score      = 74,
        status     = "Cukup Stabil",
        deductions = [{"rule": "Tabungan di bawah ideal (< 20%)", "points": -8}],
    ))

    print("\n=== TEST: narrate_forecast ===")
    print(narrator.narrate_forecast({
        "days_ahead"     : 7,
        "total_predicted": 1_250_000,
        "avg_per_day"    : 178_571,
        "peak_day_name"  : "Saturday",
        "peak_amount"    : 320_000,
    }))
