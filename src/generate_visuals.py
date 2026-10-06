"""
Ana Görselleştirme Scripti
===========================
Tüm görselleri ve PDF raporunu üretir.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI olmadan çalıştır
import matplotlib.pyplot as plt

from visualization import (
    plot_zone_heatmap,
    plot_comparison_heatmap,
    plot_event_scatter,
    generate_pdf_report,
)


def main():
    print("=" * 60)
    print("🎨 Görselleştirme Pipeline Başlıyor")
    print("=" * 60)

    # Veriyi yükle
    df = pd.read_parquet("data/processed/events_zoned_4x1.parquet")
    teams = sorted(df["team"].unique())
    print(f"\n📋 Takımlar: {', '.join(teams)}")

    # ────────────────────────────────────────
    # 1. Argentina (veri varsa) veya Spain zone heatmap
    # ────────────────────────────────────────
    # En çok olayı olan takımları bul
    team_event_counts = df.groupby("team").size().sort_values(ascending=False)
    top_teams = team_event_counts.head(4).index.tolist()
    print(f"\n🏆 En çok olaylı takımlar: {top_teams}")

    # İlk 2 takım için zone heatmap
    for team in top_teams[:2]:
        print(f"\n📊 {team} — Zone Heatmap üretiliyor...")
        safe_name = team.lower().replace(" ", "_")

        # Olay sayısı heatmap
        fig = plot_zone_heatmap(
            df, team,
            n_cols=4, n_rows=1,
            metric="count",
            save_path=f"outputs/figures/{safe_name}_zone_heatmap_count.png",
        )
        plt.close(fig)

        # Pas başarı oranı heatmap
        fig = plot_zone_heatmap(
            df, team,
            n_cols=4, n_rows=1,
            metric="pass_success",
            save_path=f"outputs/figures/{safe_name}_zone_heatmap_pass.png",
        )
        plt.close(fig)

        # Pas scatter
        fig = plot_event_scatter(
            df, team,
            event_type="Pass",
            save_path=f"outputs/figures/{safe_name}_pass_scatter.png",
        )
        plt.close(fig)

        # Şut scatter
        fig = plot_event_scatter(
            df, team,
            event_type="Shot",
            save_path=f"outputs/figures/{safe_name}_shot_scatter.png",
        )
        plt.close(fig)

    # ────────────────────────────────────────
    # 2. Karşılaştırmalı Heatmap (Top 2 takım)
    # ────────────────────────────────────────
    if len(top_teams) >= 2:
        print(f"\n🔄 Karşılaştırma: {top_teams[0]} vs {top_teams[1]}...")
        fig = plot_comparison_heatmap(
            df,
            team1=top_teams[0],
            team2=top_teams[1],
            save_path="outputs/figures/comparison_heatmap.png",
        )
        plt.close(fig)

    # ────────────────────────────────────────
    # 3. PDF Raporlar
    # ────────────────────────────────────────
    for team in top_teams[:2]:
        safe_name = team.lower().replace(" ", "_")
        print(f"\n📄 {team} — PDF rapor üretiliyor...")
        generate_pdf_report(
            df, team,
            n_cols=4, n_rows=1,
            save_path=f"outputs/reports/{safe_name}_report.pdf",
        )

    # ────────────────────────────────────────
    # 4. Özet
    # ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ Tüm görseller üretildi!")
    print("=" * 60)

    # Üretilen dosyaları listele
    for folder in ["outputs/figures", "outputs/reports"]:
        if os.path.exists(folder):
            files = os.listdir(folder)
            print(f"\n📁 {folder}/")
            for f in sorted(files):
                filepath = os.path.join(folder, f)
                size_kb = os.path.getsize(filepath) / 1024
                print(f"   📎 {f} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
