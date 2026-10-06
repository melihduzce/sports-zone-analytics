"""
Data Ingestion Module
=====================
StatsBomb açık veri setinden futbol maç verilerini çeker.
Her olay (event) x,y koordinatları ile birlikte gelir.
"""

import os
import json
import pandas as pd
from statsbombpy import sb


# ─────────────────────────────────────────────
# 1. Mevcut Yarışmaları Listele
# ─────────────────────────────────────────────
def get_competitions() -> pd.DataFrame:
    """StatsBomb açık verisindeki tüm yarışmaları döndürür."""
    competitions = sb.competitions()
    return competitions


# ─────────────────────────────────────────────
# 2. Bir Yarışmanın Maçlarını Listele
# ─────────────────────────────────────────────
def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Belirli bir yarışma ve sezon için maçları döndürür."""
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    return matches


# ─────────────────────────────────────────────
# 3. Bir Maçın Tüm Olaylarını Çek
# ─────────────────────────────────────────────
def get_match_events(match_id: int) -> pd.DataFrame:
    """
    Belirli bir maçın tüm event verilerini çeker.
    Her event'te location (x, y) bilgisi bulunur.

    StatsBomb koordinat sistemi:
    - x: 0-120 (sahanın uzunluğu, yarddan yarda)
    - y: 0-80  (sahanın genişliği)
    - (0,0) sol alt köşe
    """
    events = sb.events(match_id=match_id)
    return events


# ─────────────────────────────────────────────
# 4. Bir Yarışmadaki Tüm Maçların Olaylarını Çek
# ─────────────────────────────────────────────
def get_all_events_for_competition(
    competition_id: int, season_id: int, max_matches: int = None
) -> pd.DataFrame:
    """
    Bir yarışma-sezon çiftindeki tüm maçların event verilerini çeker.

    Args:
        competition_id: Yarışma ID
        season_id: Sezon ID
        max_matches: Opsiyonel, en fazla kaç maç çekilsin

    Returns:
        Tüm olayları içeren DataFrame
    """
    matches = get_matches(competition_id, season_id)

    if max_matches:
        matches = matches.head(max_matches)

    all_events = []
    for idx, match in matches.iterrows():
        match_id = match["match_id"]
        print(f"  Çekiliyor: {match['home_team']} vs {match['away_team']} (ID: {match_id})")
        try:
            events = get_match_events(match_id)
            events["match_id"] = match_id
            events["home_team"] = match["home_team"]
            events["away_team"] = match["away_team"]
            events["match_date"] = match.get("match_date", None)
            events["home_score"] = match.get("home_score", None)
            events["away_score"] = match.get("away_score", None)
            all_events.append(events)
        except Exception as e:
            print(f"    ⚠️ Hata: {e}")
            continue

    if all_events:
        return pd.concat(all_events, ignore_index=True)
    return pd.DataFrame()


# ─────────────────────────────────────────────
# 5. Ham Veriyi Kaydet
# ─────────────────────────────────────────────
def save_raw_data(df: pd.DataFrame, filename: str, data_dir: str = "data/raw"):
    """DataFrame'i parquet formatında kaydeder."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    df.to_parquet(filepath, index=False)
    print(f"✅ Kaydedildi: {filepath} ({len(df)} satır)")
    return filepath


# ─────────────────────────────────────────────
# Ana Çalıştırma
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🏟️  StatsBomb Veri Çekme Başlıyor")
    print("=" * 60)

    # 1. Yarışmaları listele
    print("\n📋 Mevcut yarışmalar:")
    comps = get_competitions()
    print(comps[["competition_id", "season_id", "competition_name", "season_name"]].to_string())

    # 2. FIFA World Cup 2022 verisi çek (en zengin açık veri)
    # competition_id=43 (World Cup), season_id=106 (2022)
    COMPETITION_ID = 43  # FIFA World Cup
    SEASON_ID = 106       # 2022

    print(f"\n⚽ Yarışma verileri çekiliyor (ID: {COMPETITION_ID}, Sezon: {SEASON_ID})...")
    matches = get_matches(COMPETITION_ID, SEASON_ID)
    print(f"   Toplam {len(matches)} maç bulundu.")
    save_raw_data(matches, "matches_wc2022.parquet")

    # 3. İlk 10 maçın olaylarını çek (hız için sınırlıyoruz)
    print(f"\n📊 Maç olayları çekiliyor (ilk 10 maç)...")
    events = get_all_events_for_competition(COMPETITION_ID, SEASON_ID, max_matches=10)
    if not events.empty:
        save_raw_data(events, "events_wc2022_sample.parquet")
        print(f"\n📈 Toplam {len(events)} olay çekildi.")
        print(f"   Olay tipleri: {events['type'].nunique()} farklı tip")
        print(f"   Lokasyon verisi olan: {events['location'].notna().sum()} olay")
