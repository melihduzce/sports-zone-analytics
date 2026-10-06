"""
Visualization Module
=====================
Zone bazlı heatmap'ler, radar chart'lar ve PDF raporlar üretir.
mplsoccer kütüphanesi ile profesyonel futbol sahası görselleri oluşturur.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

try:
    from mplsoccer import Pitch, VerticalPitch
    HAS_MPLSOCCER = True
except ImportError:
    HAS_MPLSOCCER = False
    print("⚠️ mplsoccer yüklü değil. Basit görselleştirme kullanılacak.")


# ─────────────────────────────────────────────
# Renk Paleti
# ─────────────────────────────────────────────
ZONE_CMAP = "YlOrRd"  # Sarı → Turuncu → Kırmızı


# ─────────────────────────────────────────────
# 1. Zone Heatmap (Saha Üzerinde)
# ─────────────────────────────────────────────
def plot_zone_heatmap(
    df: pd.DataFrame,
    team: str,
    n_cols: int = 4,
    n_rows: int = 1,
    metric: str = "count",
    title: str = None,
    save_path: str = None,
    figsize: tuple = (14, 9),
) -> plt.Figure:
    """
    Futbol sahası üzerinde zone bazlı heatmap çizer.

    Args:
        df: Zone bilgisi içeren event DataFrame
        team: Analiz edilecek takım adı
        n_cols: Grid sütun sayısı
        n_rows: Grid satır sayısı
        metric: 'count' (olay sayısı) veya 'pass_success' (pas başarı %)
        title: Grafik başlığı
        save_path: Kaydedilecek dosya yolu (PNG/JPEG)
    """
    team_df = df[df["team"] == team].copy()

    # Zone bazlı metrik hesapla
    if metric == "count":
        zone_values = team_df.groupby("zone").size()
        metric_label = "Olay Sayısı"
    elif metric == "pass_success":
        passes = team_df[team_df["type"] == "Pass"]
        zone_values = passes.groupby("zone").apply(
            lambda x: x["pass_outcome"].isna().mean() * 100 if len(x) > 0 else 0
        )
        metric_label = "Pas Başarı Oranı (%)"
    else:
        zone_values = team_df.groupby("zone").size()
        metric_label = "Olay Sayısı"

    # Pitch boyutları
    pitch_length = 120
    pitch_width = 80
    x_step = pitch_length / n_cols
    y_step = pitch_width / n_rows

    if HAS_MPLSOCCER:
        # mplsoccer ile profesyonel saha çizimi
        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color="#1a472a",
            line_color="white",
            line_zorder=2,
            linewidth=1.5,
        )
        fig, ax = pitch.draw(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, pitch_length)
        ax.set_ylim(0, pitch_width)
        ax.set_facecolor("#1a472a")
        ax.set_aspect("equal")

    # Zone dikdörtgenlerini çiz
    max_val = zone_values.max() if len(zone_values) > 0 else 1
    min_val = zone_values.min() if len(zone_values) > 0 else 0
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.colormaps[ZONE_CMAP]

    for zone_num in range(1, n_cols * n_rows + 1):
        # Zone koordinatlarını hesapla
        col_idx = (zone_num - 1) % n_cols
        row_idx = (zone_num - 1) // n_cols

        x_start = col_idx * x_step
        y_start = row_idx * y_step

        value = zone_values.get(zone_num, 0)
        color = cmap(norm(value)) if value > 0 else (0.1, 0.3, 0.1, 0.3)

        # Dikdörtgen çiz
        rect = plt.Rectangle(
            (x_start, y_start), x_step, y_step,
            facecolor=color, edgecolor="white",
            linewidth=2, alpha=0.7, zorder=1,
        )
        ax.add_patch(rect)

        # Zone numarası ve değer yazısı
        center_x = x_start + x_step / 2
        center_y = y_start + y_step / 2

        ax.text(
            center_x, center_y + 3,
            f"Zone {zone_num}",
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color="white", zorder=3,
        )
        ax.text(
            center_x, center_y - 5,
            f"{value:.0f}" if metric == "count" else f"{value:.1f}%",
            ha="center", va="center",
            fontsize=18, fontweight="bold",
            color="white", zorder=3,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
        )

    # Renk skalası
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(metric_label, fontsize=12, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Başlık
    if title is None:
        title = f"{team} — Zone Heatmap ({metric_label})"
    ax.set_title(title, fontsize=18, fontweight="bold", color="white", pad=15)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"📸 Kaydedildi: {save_path}")

    return fig


# ─────────────────────────────────────────────
# 2. Karşılaştırmalı Heatmap (2 Takım)
# ─────────────────────────────────────────────
def plot_comparison_heatmap(
    df: pd.DataFrame,
    team1: str,
    team2: str,
    n_cols: int = 4,
    n_rows: int = 1,
    save_path: str = None,
) -> plt.Figure:
    """İki takımın zone kullanımını yan yana karşılaştırır."""
    fig, axes = plt.subplots(1, 2, figsize=(24, 9))

    for ax, team in zip(axes, [team1, team2]):
        team_df = df[df["team"] == team]
        zone_counts = team_df.groupby("zone").size()
        total = zone_counts.sum()
        zone_pcts = (zone_counts / total * 100).fillna(0)

        pitch_length, pitch_width = 120, 80
        x_step = pitch_length / n_cols
        y_step = pitch_width / n_rows

        ax.set_xlim(0, pitch_length)
        ax.set_ylim(0, pitch_width)
        ax.set_facecolor("#1a472a")
        ax.set_aspect("equal")

        max_val = zone_pcts.max() if len(zone_pcts) > 0 else 1
        norm = mcolors.Normalize(vmin=0, vmax=max_val)
        cmap = plt.colormaps[ZONE_CMAP]

        for zone_num in range(1, n_cols * n_rows + 1):
            col_idx = (zone_num - 1) % n_cols
            row_idx = (zone_num - 1) // n_cols
            x_start = col_idx * x_step
            y_start = row_idx * y_step

            value = zone_pcts.get(zone_num, 0)
            color = cmap(norm(value)) if value > 0 else (0.1, 0.3, 0.1, 0.3)

            rect = plt.Rectangle(
                (x_start, y_start), x_step, y_step,
                facecolor=color, edgecolor="white",
                linewidth=2, alpha=0.7,
            )
            ax.add_patch(rect)

            center_x = x_start + x_step / 2
            center_y = y_start + y_step / 2
            ax.text(center_x, center_y + 3, f"Zone {zone_num}",
                    ha="center", va="center", fontsize=12, fontweight="bold", color="white")
            ax.text(center_x, center_y - 5, f"%{value:.1f}",
                    ha="center", va="center", fontsize=16, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5))

        ax.set_title(team, fontsize=16, fontweight="bold", color="white", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Zone Kullanım Karşılaştırması (%)", fontsize=20,
                 fontweight="bold", color="white", y=1.02)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"📸 Kaydedildi: {save_path}")

    return fig


# ─────────────────────────────────────────────
# 3. Event Scatter (Saha Üzerinde Noktalar)
# ─────────────────────────────────────────────
def plot_event_scatter(
    df: pd.DataFrame,
    team: str,
    event_type: str = "Pass",
    save_path: str = None,
) -> plt.Figure:
    """Bir takımın belirli olay tipini saha üzerinde scatter plot olarak gösterir."""
    team_events = df[(df["team"] == team) & (df["type"] == event_type)]

    if HAS_MPLSOCCER:
        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color="#1a472a",
            line_color="white",
        )
        fig, ax = pitch.draw(figsize=(14, 9))
    else:
        fig, ax = plt.subplots(figsize=(14, 9))
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 80)
        ax.set_facecolor("#1a472a")
        ax.set_aspect("equal")

    # Başarılı / Başarısız olayları ayır
    if event_type == "Pass" and "pass_outcome" in team_events.columns:
        successful = team_events[team_events["pass_outcome"].isna()]
        failed = team_events[team_events["pass_outcome"].notna()]

        ax.scatter(successful["loc_x"], successful["loc_y"],
                   c="#00ff88", alpha=0.4, s=20, label=f"Başarılı ({len(successful)})", zorder=2)
        ax.scatter(failed["loc_x"], failed["loc_y"],
                   c="#ff4444", alpha=0.4, s=20, label=f"Başarısız ({len(failed)})", zorder=2)
    else:
        ax.scatter(team_events["loc_x"], team_events["loc_y"],
                   c="#00aaff", alpha=0.4, s=20, label=f"{event_type} ({len(team_events)})", zorder=2)

    ax.legend(loc="upper left", fontsize=11, facecolor="black", edgecolor="white",
              labelcolor="white")
    ax.set_title(f"{team} — {event_type} Dağılımı", fontsize=16, fontweight="bold",
                 color="white", pad=15)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"📸 Kaydedildi: {save_path}")

    return fig


# ─────────────────────────────────────────────
# 4. PDF Rapor
# ─────────────────────────────────────────────
def generate_pdf_report(
    df: pd.DataFrame,
    team: str,
    n_cols: int = 4,
    n_rows: int = 1,
    save_path: str = "outputs/reports/team_report.pdf",
):
    """
    Bir takım için tek dosyada PDF rapor üretir.
    İçerik: Zone heatmap + Pass scatter + Shot scatter
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with PdfPages(save_path) as pdf:
        # Sayfa 1: Zone Heatmap (Olay Sayısı)
        fig1 = plot_zone_heatmap(df, team, n_cols=n_cols, n_rows=n_rows, metric="count")
        pdf.savefig(fig1, facecolor=fig1.get_facecolor())
        plt.close(fig1)

        # Sayfa 2: Zone Heatmap (Pas Başarı Oranı)
        fig2 = plot_zone_heatmap(df, team, n_cols=n_cols, n_rows=n_rows, metric="pass_success")
        pdf.savefig(fig2, facecolor=fig2.get_facecolor())
        plt.close(fig2)

        # Sayfa 3: Pas Dağılımı
        fig3 = plot_event_scatter(df, team, event_type="Pass")
        pdf.savefig(fig3, facecolor=fig3.get_facecolor())
        plt.close(fig3)

        # Sayfa 4: Şut Dağılımı
        fig4 = plot_event_scatter(df, team, event_type="Shot")
        pdf.savefig(fig4, facecolor=fig4.get_facecolor())
        plt.close(fig4)

    print(f"📄 PDF rapor oluşturuldu: {save_path}")
