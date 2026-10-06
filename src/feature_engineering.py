"""
Feature Engineering Module
===========================
Zone bazlı özellik çıkarma. Her takım-maç çifti için
zone bazlı metrikler hesaplar.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 1. Zone Bazlı Temel Metrikler
# ─────────────────────────────────────────────
def compute_zone_metrics(df: pd.DataFrame, team_col: str = "team") -> pd.DataFrame:
    """
    Her takım ve zone kombinasyonu için temel metrikleri hesaplar.

    Çıktı kolonları:
        - total_events: Toplam olay sayısı
        - pass_count: Pas sayısı
        - pass_success_rate: Başarılı pas oranı
        - shot_count: Şut sayısı
        - carry_count: Top taşıma sayısı
        - pressure_count: Baskı sayısı
        - dribble_count: Çalım sayısı
        - event_pct: O zone'daki olayların toplam olaylara oranı
    """
    results = []

    for team in df[team_col].unique():
        team_df = df[df[team_col] == team]
        total_team_events = len(team_df)

        for zone in sorted(team_df["zone"].dropna().unique()):
            zone_df = team_df[team_df["zone"] == zone]

            # Pas metrikleri
            passes = zone_df[zone_df["type"] == "Pass"]
            pass_count = len(passes)
            if pass_count > 0 and "pass_outcome" in passes.columns:
                # StatsBomb'da başarılı paslarda pass_outcome NaN olur
                pass_success = passes["pass_outcome"].isna().sum()
                pass_success_rate = pass_success / pass_count
            else:
                pass_success_rate = np.nan

            # Şut metrikleri
            shots = zone_df[zone_df["type"] == "Shot"]
            shot_count = len(shots)

            # xG (beklenen gol)
            xg_total = 0.0
            if shot_count > 0 and "shot_statsbomb_xg" in shots.columns:
                xg_total = shots["shot_statsbomb_xg"].sum()

            # Diğer metrikler
            carry_count = len(zone_df[zone_df["type"] == "Carry"])
            pressure_count = len(zone_df[zone_df["type"] == "Pressure"])
            dribble_count = len(zone_df[zone_df["type"] == "Dribble"])

            results.append({
                "team": team,
                "zone": int(zone),
                "zone_label": zone_df["zone_label"].iloc[0] if "zone_label" in zone_df.columns else f"Zone {int(zone)}",
                "total_events": len(zone_df),
                "event_pct": len(zone_df) / total_team_events * 100 if total_team_events > 0 else 0,
                "pass_count": pass_count,
                "pass_success_rate": round(pass_success_rate, 3) if not np.isnan(pass_success_rate) else np.nan,
                "shot_count": shot_count,
                "xg_total": round(xg_total, 3),
                "carry_count": carry_count,
                "pressure_count": pressure_count,
                "dribble_count": dribble_count,
            })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 2. Maç Bazlı Zone Profili
# ─────────────────────────────────────────────
def compute_match_zone_profiles(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her maç-takım çifti için zone profilini hesaplar.
    Bu, takımların maçtan maça nasıl değiştiğini analiz etmek için kullanılır.
    """
    results = []

    for match_id in df["match_id"].unique():
        match_df = df[df["match_id"] == match_id]

        for team in match_df["team"].unique():
            team_df = match_df[match_df["team"] == team]
            total = len(team_df)

            profile = {
                "match_id": match_id,
                "team": team,
                "home_team": team_df["home_team"].iloc[0] if "home_team" in team_df.columns else None,
                "away_team": team_df["away_team"].iloc[0] if "away_team" in team_df.columns else None,
            }

            # Her zone için olay yüzdesi
            for zone in sorted(team_df["zone"].dropna().unique()):
                zone_count = len(team_df[team_df["zone"] == zone])
                profile[f"zone_{int(zone)}_pct"] = round(zone_count / total * 100, 2) if total > 0 else 0

            results.append(profile)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 3. Takım Genel Profili (Sezon Bazlı)
# ─────────────────────────────────────────────
def compute_team_season_profile(zone_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Zone metriklerinden takım bazlı pivot tablo oluşturur.
    Her satır bir takım, her kolon bir zone metriği.
    Clustering için input olarak kullanılır.
    """
    # Her takım için zone bazlı event yüzdesini pivot et
    pivot = zone_metrics.pivot_table(
        index="team",
        columns="zone",
        values="event_pct",
        aggfunc="mean",
    ).fillna(0)

    pivot.columns = [f"zone_{int(c)}_pct" for c in pivot.columns]

    # Ek metrikler ekle
    for zone in zone_metrics["zone"].unique():
        zone_data = zone_metrics[zone_metrics["zone"] == zone]
        agg = zone_data.groupby("team").agg(
            pass_success_rate=("pass_success_rate", "mean"),
            xg_per_match=("xg_total", "mean"),
        )
        agg.columns = [f"zone_{int(zone)}_{c}" for c in agg.columns]
        pivot = pivot.join(agg, how="left")

    return pivot.fillna(0)


if __name__ == "__main__":
    import os

    processed_path = "data/processed/events_zoned_4x1.parquet"
    if os.path.exists(processed_path):
        print("📂 İşlenmiş veri yükleniyor...")
        df = pd.read_parquet(processed_path)

        # Zone metrikleri
        print("\n📊 Zone metrikleri hesaplanıyor...")
        metrics = compute_zone_metrics(df)
        print(metrics.to_string())

        # Kaydet
        os.makedirs("data/features", exist_ok=True)
        metrics.to_parquet("data/features/zone_metrics.parquet", index=False)
        print(f"\n💾 Zone metrikleri kaydedildi: data/features/zone_metrics.parquet")

        # Maç bazlı profiller
        print("\n📊 Maç bazlı zone profilleri hesaplanıyor...")
        match_profiles = compute_match_zone_profiles(df)
        match_profiles.to_parquet("data/features/match_zone_profiles.parquet", index=False)
        print(f"💾 Maç profilleri kaydedildi: data/features/match_zone_profiles.parquet")
    else:
        print(f"❌ İşlenmiş veri bulunamadı: {processed_path}")
        print("   Önce preprocessing.py'yi çalıştırın.")
