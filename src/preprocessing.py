"""
Preprocessing & Zone Mapping Module
====================================
Ham event verisini temizler ve sahayı zone'lara bölerek
her olayı bir zone'a atar.

StatsBomb Koordinat Sistemi:
    x: 0 → 120  (sahanın uzunluğu)
    y: 0 → 80   (sahanın genişliği)
    Takımlar her yarıda kendi kalesinden rakip kaleye doğru oynar.
    StatsBomb verisi zaten takım perspektifine göre normalize edilmiştir.
"""

import pandas as pd
import numpy as np
from typing import Tuple


# ─────────────────────────────────────────────
# Saha Boyutları (StatsBomb standardı)
# ─────────────────────────────────────────────
PITCH_LENGTH = 120
PITCH_WIDTH = 80


# ─────────────────────────────────────────────
# 1. Lokasyon Kolonlarını Ayrıştır
# ─────────────────────────────────────────────
def extract_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    StatsBomb'un 'location' kolonunu (liste formatı [x, y]) ayrıştırarak
    'loc_x' ve 'loc_y' kolonlarına çevirir.
    """
    df = df.copy()

    # location kolonu [x, y] listesi olarak geliyor
    def safe_extract(loc, idx):
        try:
            if isinstance(loc, (list, np.ndarray)) and len(loc) > idx:
                return float(loc[idx])
        except (TypeError, ValueError):
            pass
        return np.nan

    df["loc_x"] = df["location"].apply(lambda loc: safe_extract(loc, 0))
    df["loc_y"] = df["location"].apply(lambda loc: safe_extract(loc, 1))

    return df


# ─────────────────────────────────────────────
# 2. Zone Mapping (Grid Bazlı)
# ─────────────────────────────────────────────
def assign_zones(
    df: pd.DataFrame,
    n_cols: int = 4,
    n_rows: int = 1,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> pd.DataFrame:
    """
    Sahayı n_cols x n_rows grid'e böler ve her olayı bir zone'a atar.

    Varsayılan 4x1 grid (sahayı 4 dikey şeride böler):
        Zone 1: x ∈ [0, 30)   → Kendi ceza alanı / defansif
        Zone 2: x ∈ [30, 60)  → Defansif orta saha
        Zone 3: x ∈ [60, 90)  → Hücum orta saha
        Zone 4: x ∈ [90, 120] → Rakip ceza alanı / hücum

    6'lı grid (3x2):
        Sahayı 3 dikey x 2 yatay = 6 zone'a böler

    8'li grid (4x2):
        Sahayı 4 dikey x 2 yatay = 8 zone'a böler

    Args:
        df: loc_x ve loc_y kolonları olan DataFrame
        n_cols: Dikey bölme sayısı (x ekseni boyunca)
        n_rows: Yatay bölme sayısı (y ekseni boyunca)

    Returns:
        'zone' kolonu eklenmiş DataFrame
    """
    df = df.copy()

    # Zone sınırlarını hesapla
    x_bins = np.linspace(0, pitch_length, n_cols + 1)
    y_bins = np.linspace(0, pitch_width, n_rows + 1)

    # Her koordinatı bin'e ata (0-indexed)
    x_zone = np.digitize(df["loc_x"].fillna(-1), x_bins, right=False) - 1
    y_zone = np.digitize(df["loc_y"].fillna(-1), y_bins, right=False) - 1

    # Sınır düzeltmeleri (120 ve 80 tam sınırda olanlar)
    x_zone = np.clip(x_zone, 0, n_cols - 1)
    y_zone = np.clip(y_zone, 0, n_rows - 1)

    # Tek bir zone numarası oluştur (1-indexed)
    df["zone"] = (y_zone * n_cols + x_zone + 1).astype(int)

    # Lokasyonu olmayan olaylara NaN ata
    mask_no_location = df["loc_x"].isna() | df["loc_y"].isna()
    df.loc[mask_no_location, "zone"] = np.nan

    return df


# ─────────────────────────────────────────────
# 3. Zone Etiketleri
# ─────────────────────────────────────────────
def get_zone_labels(n_cols: int = 4, n_rows: int = 1) -> dict:
    """
    Zone numaralarına açıklayıcı etiketler atar.
    """
    if n_cols == 4 and n_rows == 1:
        return {
            1: "Defansif Alan",
            2: "Defansif Orta Saha",
            3: "Hücum Orta Saha",
            4: "Hücum Alanı",
        }
    elif n_cols == 4 and n_rows == 2:
        return {
            1: "Defansif Sol",
            2: "Defansif Orta Sol",
            3: "Hücum Orta Sol",
            4: "Hücum Sol",
            5: "Defansif Sağ",
            6: "Defansif Orta Sağ",
            7: "Hücum Orta Sağ",
            8: "Hücum Sağ",
        }
    else:
        # Genel durumda numaralı etiketler
        return {i: f"Zone {i}" for i in range(1, n_cols * n_rows + 1)}


# ─────────────────────────────────────────────
# 4. Olay Tiplerini Filtrele
# ─────────────────────────────────────────────
def filter_event_types(
    df: pd.DataFrame,
    event_types: list = None,
) -> pd.DataFrame:
    """
    Belirli olay tiplerini filtreler.

    Yaygın olay tipleri:
        - Pass: Paslar
        - Shot: Şutlar
        - Carry: Top taşıma
        - Pressure: Baskı
        - Dribble: Çalım
        - Ball Receipt*: Top alma
        - Duel: İkili mücadele
        - Interception: Top kesme
    """
    if event_types is None:
        # Varsayılan: top ile yapılan ana olaylar
        event_types = ["Pass", "Shot", "Carry", "Dribble", "Ball Receipt*", "Pressure"]

    return df[df["type"].isin(event_types)].copy()


# ─────────────────────────────────────────────
# 5. Tam Preprocessing Pipeline
# ─────────────────────────────────────────────
def preprocess_events(
    df: pd.DataFrame,
    n_cols: int = 4,
    n_rows: int = 1,
    event_types: list = None,
) -> pd.DataFrame:
    """
    Tam preprocessing pipeline:
    1. Koordinatları ayrıştır
    2. Olay tiplerini filtrele
    3. Zone'lara ata
    4. Zone etiketlerini ekle
    """
    print(f"🔧 Preprocessing başlıyor ({len(df)} olay)...")

    # 1. Koordinat çıkarma
    df = extract_coordinates(df)
    loc_count = df["loc_x"].notna().sum()
    print(f"   📍 Lokasyon verisi olan: {loc_count} olay")

    # 2. Olay filtresi
    if event_types:
        df = filter_event_types(df, event_types)
        print(f"   🎯 Filtrelenen olay tipleri: {event_types}")
    else:
        # Lokasyonu olan tüm olayları tut
        df = df[df["loc_x"].notna()].copy()

    print(f"   📊 Filtreleme sonrası: {len(df)} olay")

    # 3. Zone mapping
    df = assign_zones(df, n_cols=n_cols, n_rows=n_rows)

    # 4. Zone etiketleri
    labels = get_zone_labels(n_cols, n_rows)
    df["zone_label"] = df["zone"].map(labels)

    print(f"   ✅ Zone mapping tamamlandı ({n_cols}x{n_rows} grid)")
    print(f"   Zone dağılımı:")
    zone_dist = df["zone"].value_counts().sort_index()
    for zone, count in zone_dist.items():
        label = labels.get(int(zone), f"Zone {int(zone)}")
        print(f"      Zone {int(zone)} ({label}): {count} olay")

    return df


if __name__ == "__main__":
    import os

    # İşlenmiş veri varsa yükle
    raw_path = "data/raw/events_wc2022_sample.parquet"
    if os.path.exists(raw_path):
        print("📂 Ham veri yükleniyor...")
        df = pd.read_parquet(raw_path)

        # 4'lü grid ile işle
        processed = preprocess_events(df, n_cols=4, n_rows=1)

        # Kaydet
        os.makedirs("data/processed", exist_ok=True)
        processed.to_parquet("data/processed/events_zoned_4x1.parquet", index=False)
        print(f"\n💾 İşlenmiş veri kaydedildi: data/processed/events_zoned_4x1.parquet")
    else:
        print(f"❌ Ham veri bulunamadı: {raw_path}")
        print("   Önce data_ingestion.py'yi çalıştırın.")
