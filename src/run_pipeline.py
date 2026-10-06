"""
Full Pipeline Runner
=====================
Tüm adımları sırasıyla çalıştırır:
1. Tüm WC 2022 maçlarının verisini çek (64 maç)
2. Zone mapping
3. Feature engineering
4. Clustering
5. Predictive model
6. MLflow logging

Kullanım:
    python src/run_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")

from data_ingestion import get_all_events_for_competition, save_raw_data
from preprocessing import preprocess_events
from feature_engineering import compute_zone_metrics, compute_match_zone_profiles, compute_team_season_profile
from models import find_optimal_k, run_clustering, prepare_match_features, train_predictive_model, run_mlflow_experiment
from visualization import plot_zone_heatmap, plot_comparison_heatmap, generate_pdf_report

import pandas as pd
import matplotlib.pyplot as plt


def main():
    COMPETITION_ID = 43   # FIFA World Cup
    SEASON_ID = 106        # 2022
    N_COLS = 4
    N_ROWS = 1

    print("=" * 70)
    print("🏟️  SPORTS ZONE ANALYTICS — FULL PIPELINE")
    print("=" * 70)

    # ═══════════════════════════════════════════
    # STEP 1: Veri Çekme (Tüm 64 maç)
    # ═══════════════════════════════════════════
    raw_path = "data/raw/events_wc2022_full.parquet"

    if os.path.exists(raw_path):
        print(f"\n📂 Veri zaten mevcut, yükleniyor: {raw_path}")
        events_raw = pd.read_parquet(raw_path)
    else:
        print(f"\n⚽ [STEP 1] Tüm WC 2022 maçlarının verileri çekiliyor...")
        events_raw = get_all_events_for_competition(COMPETITION_ID, SEASON_ID, max_matches=None)
        save_raw_data(events_raw, "events_wc2022_full.parquet")

    print(f"   Toplam olay: {len(events_raw)}")
    print(f"   Maç sayısı: {events_raw['match_id'].nunique()}")

    # ═══════════════════════════════════════════
    # STEP 2: Preprocessing & Zone Mapping
    # ═══════════════════════════════════════════
    print(f"\n🔧 [STEP 2] Zone mapping ({N_COLS}x{N_ROWS} grid)...")
    events_zoned = preprocess_events(events_raw, n_cols=N_COLS, n_rows=N_ROWS)

    os.makedirs("data/processed", exist_ok=True)
    events_zoned.to_parquet("data/processed/events_zoned_full.parquet", index=False)

    # ═══════════════════════════════════════════
    # STEP 3: Feature Engineering
    # ═══════════════════════════════════════════
    print(f"\n📊 [STEP 3] Feature engineering...")
    zone_metrics = compute_zone_metrics(events_zoned)

    os.makedirs("data/features", exist_ok=True)
    zone_metrics.to_parquet("data/features/zone_metrics_full.parquet", index=False)

    # Takım sezon profili (clustering input)
    team_profiles = compute_team_season_profile(zone_metrics)
    team_profiles.to_parquet("data/features/team_profiles.parquet")
    print(f"   Takım profili: {team_profiles.shape[0]} takım, {team_profiles.shape[1]} özellik")
    print(f"   Takımlar: {', '.join(team_profiles.index.tolist())}")

    # ═══════════════════════════════════════════
    # STEP 4a: Top Takımlar Heatmap (güncelle)
    # ═══════════════════════════════════════════
    print(f"\n🎨 [STEP 4a] Görselleştirmeler güncelleniyor...")
    team_event_counts = events_zoned.groupby("team").size().sort_values(ascending=False)
    top_teams = team_event_counts.head(4).index.tolist()

    for team in top_teams[:2]:
        safe_name = team.lower().replace(" ", "_")
        fig = plot_zone_heatmap(events_zoned, team, n_cols=N_COLS, n_rows=N_ROWS,
                                save_path=f"outputs/figures/{safe_name}_zone_heatmap_full.png")
        plt.close(fig)

    if len(top_teams) >= 2:
        fig = plot_comparison_heatmap(events_zoned, top_teams[0], top_teams[1],
                                      save_path="outputs/figures/comparison_full.png")
        plt.close(fig)

    # ═══════════════════════════════════════════
    # STEP 4b: Clustering
    # ═══════════════════════════════════════════
    print(f"\n🔬 [STEP 4b] K-Means Clustering...")

    # Optimal K bul
    zone_pct_cols = [c for c in team_profiles.columns if c.endswith("_pct")]
    cluster_features = team_profiles[zone_pct_cols]

    optimal = find_optimal_k(
        cluster_features,
        k_range=range(2, min(8, len(cluster_features))),
        save_path="outputs/figures/optimal_k.png",
    )

    # Clustering çalıştır
    clustering_results = run_clustering(
        cluster_features,
        n_clusters=optimal["best_k"],
        save_path="outputs/figures/cluster_profiles.png",
    )

    # ═══════════════════════════════════════════
    # STEP 4c: Predictive Model
    # ═══════════════════════════════════════════
    print(f"\n🤖 [STEP 4c] Predictive Model (Maç Sonucu Tahmini)...")

    match_features = prepare_match_features(events_zoned, n_cols=N_COLS, n_rows=N_ROWS)
    match_features.to_parquet("data/features/match_features.parquet", index=False)

    prediction_results = train_predictive_model(
        match_features,
        save_path="outputs/figures/model_comparison.png",
    )

    # ═══════════════════════════════════════════
    # STEP 5: PDF Raporlar (güncelle)
    # ═══════════════════════════════════════════
    print(f"\n📄 [STEP 5] PDF raporlar güncelleniyor...")
    for team in top_teams[:2]:
        safe_name = team.lower().replace(" ", "_")
        generate_pdf_report(events_zoned, team, n_cols=N_COLS, n_rows=N_ROWS,
                           save_path=f"outputs/reports/{safe_name}_full_report.pdf")

    # ═══════════════════════════════════════════
    # STEP 6: MLflow
    # ═══════════════════════════════════════════
    print(f"\n⚙️  [STEP 6] MLflow experiment tracking...")
    run_mlflow_experiment(clustering_results, prediction_results, team_profiles)

    # ═══════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("✅ FULL PIPELINE TAMAMLANDI!")
    print("=" * 70)

    print(f"\n📊 Özet:")
    print(f"   Toplam olay: {len(events_zoned)}")
    print(f"   Takım sayısı: {events_zoned['team'].nunique()}")
    print(f"   Maç sayısı: {events_zoned['match_id'].nunique()}")
    print(f"   Clustering: K={optimal['best_k']}, Silhouette={clustering_results['silhouette_score']:.3f}")
    print(f"   Best Model: {prediction_results['best_model_name']}")
    print(f"   CV Accuracy: RF={prediction_results['rf_cv_mean']:.3f}, XGB={prediction_results['xgb_cv_mean']:.3f}")

    # Dosyaları listele
    for folder in ["outputs/figures", "outputs/reports"]:
        if os.path.exists(folder):
            files = os.listdir(folder)
            print(f"\n📁 {folder}/")
            for f in sorted(files):
                filepath = os.path.join(folder, f)
                size_kb = os.path.getsize(filepath) / 1024
                print(f"   📎 {f} ({size_kb:.1f} KB)")

    print(f"\n🔗 MLflow UI başlatmak için:")
    print(f"   cd sports-zone-analytics && mlflow ui --backend-store-uri mlruns")


if __name__ == "__main__":
    main()
