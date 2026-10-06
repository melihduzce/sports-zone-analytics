"""
Models Module
==============
1. K-Means Clustering: Takımları zone kullanım paternlerine göre gruplar
2. Predictive Model: Zone paterninden maç sonucunu tahmin eder

MLOps öğrenme hedefleri:
- Unsupervised vs Supervised ML farkı
- Feature scaling, model selection, hyperparameter tuning
- Cross-validation, metrik seçimi
- MLflow ile experiment tracking
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


# ═══════════════════════════════════════════════
#  PART 1: K-MEANS CLUSTERING
# ═══════════════════════════════════════════════

def find_optimal_k(
    features: pd.DataFrame,
    k_range: range = range(2, 8),
    save_path: str = None,
) -> dict:
    """
    Elbow method + Silhouette score ile optimal küme sayısını bulur.

    MLOps öğrenme noktası:
    - Hyperparameter search (basit grid search)
    - Metrik bazlı model seçimi
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    results = {"k": [], "inertia": [], "silhouette": []}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        results["k"].append(k)
        results["inertia"].append(kmeans.inertia_)
        results["silhouette"].append(silhouette_score(X_scaled, labels))

        print(f"   K={k}: Inertia={kmeans.inertia_:.1f}, Silhouette={results['silhouette'][-1]:.3f}")

    # En iyi K (en yüksek silhouette)
    best_idx = np.argmax(results["silhouette"])
    best_k = results["k"][best_idx]
    print(f"\n   🏆 Optimal K = {best_k} (Silhouette = {results['silhouette'][best_idx]:.3f})")

    # Elbow + Silhouette grafiği
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    # Elbow
    ax1.plot(results["k"], results["inertia"], "o-", color="#00aaff", linewidth=2, markersize=8)
    ax1.axvline(x=best_k, color="#ff4444", linestyle="--", alpha=0.7, label=f"Optimal K={best_k}")
    ax1.set_xlabel("K (Küme Sayısı)", fontsize=12, color="white")
    ax1.set_ylabel("Inertia (WCSS)", fontsize=12, color="white")
    ax1.set_title("Elbow Method", fontsize=14, fontweight="bold", color="white")
    ax1.set_facecolor("#1a1a2e")
    ax1.tick_params(colors="white")
    ax1.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")

    # Silhouette
    ax2.plot(results["k"], results["silhouette"], "o-", color="#00ff88", linewidth=2, markersize=8)
    ax2.axvline(x=best_k, color="#ff4444", linestyle="--", alpha=0.7, label=f"Optimal K={best_k}")
    ax2.set_xlabel("K (Küme Sayısı)", fontsize=12, color="white")
    ax2.set_ylabel("Silhouette Score", fontsize=12, color="white")
    ax2.set_title("Silhouette Analysis", fontsize=14, fontweight="bold", color="white")
    ax2.set_facecolor("#1a1a2e")
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"   📸 Kaydedildi: {save_path}")

    plt.close(fig)

    return {"best_k": best_k, "results": results}


def run_clustering(
    features: pd.DataFrame,
    n_clusters: int = 3,
    save_path: str = None,
) -> dict:
    """
    K-Means clustering çalıştırır ve sonuçları görselleştirir.

    Returns:
        dict: labels, cluster_centers, silhouette_score, model, scaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels)

    # Sonuç DataFrame
    result_df = features.copy()
    result_df["cluster"] = labels
    result_df["team"] = features.index

    print(f"\n   📊 Clustering Sonuçları (K={n_clusters}):")
    print(f"   Silhouette Score: {sil_score:.3f}")
    for c in range(n_clusters):
        cluster_teams = result_df[result_df["cluster"] == c]["team"].tolist()
        print(f"   Küme {c}: {', '.join(cluster_teams)}")

    # Cluster profilleri görselleştirme
    fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 5))
    fig.patch.set_facecolor("#0d1117")

    if n_clusters == 1:
        axes = [axes]

    zone_cols = [c for c in features.columns if c.startswith("zone_") and c.endswith("_pct")]

    # Renk paleti
    cluster_colors = ["#00aaff", "#ff6b6b", "#00ff88", "#ffaa00", "#ff44ff"]

    for c, ax in enumerate(axes):
        cluster_data = features.loc[result_df[result_df["cluster"] == c].index]
        means = cluster_data[zone_cols].mean()

        bars = ax.bar(
            range(len(zone_cols)),
            means.values,
            color=cluster_colors[c % len(cluster_colors)],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_xticks(range(len(zone_cols)))
        ax.set_xticklabels([c.replace("zone_", "Z").replace("_pct", "") for c in zone_cols],
                           fontsize=10, color="white")
        ax.set_ylabel("Olay Yüzdesi (%)", fontsize=10, color="white")
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")

        cluster_teams = result_df[result_df["cluster"] == c]["team"].tolist()
        title_teams = ", ".join(cluster_teams[:3])
        if len(cluster_teams) > 3:
            title_teams += f" +{len(cluster_teams)-3}"
        ax.set_title(f"Küme {c}\n{title_teams}", fontsize=11, fontweight="bold", color="white")

        # Bar değerlerini yaz
        for bar, val in zip(bars, means.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="white")

    fig.suptitle(f"Küme Profilleri — Zone Kullanım Dağılımı (K={n_clusters}, Silhouette={sil_score:.3f})",
                 fontsize=14, fontweight="bold", color="white", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"   📸 Kaydedildi: {save_path}")

    plt.close(fig)

    return {
        "labels": labels,
        "result_df": result_df,
        "silhouette_score": sil_score,
        "model": kmeans,
        "scaler": scaler,
        "n_clusters": n_clusters,
    }


# ═══════════════════════════════════════════════
#  PART 2: PREDICTIVE MODEL (Maç Sonucu Tahmini)
# ═══════════════════════════════════════════════

def prepare_match_features(
    events_df: pd.DataFrame,
    n_cols: int = 4,
    n_rows: int = 1,
) -> pd.DataFrame:
    """
    Her maç-takım çifti için zone bazlı feature vektörü oluşturur.
    Target: Kazanan mı, kaybeden mi, berabere mi (W/L/D)

    MLOps öğrenme noktası:
    - Feature engineering for supervised learning
    - Label encoding
    """
    records = []

    for match_id in events_df["match_id"].unique():
        match_df = events_df[events_df["match_id"] == match_id]

        home_team = match_df["home_team"].iloc[0]
        away_team = match_df["away_team"].iloc[0]
        home_score = match_df["home_score"].iloc[0] if "home_score" in match_df.columns else None
        away_score = match_df["away_score"].iloc[0] if "away_score" in match_df.columns else None

        for team in match_df["team"].unique():
            team_df = match_df[match_df["team"] == team]
            total = len(team_df)
            if total == 0:
                continue

            record = {
                "match_id": match_id,
                "team": team,
            }

            # Zone yüzdeleri
            for zone in range(1, n_cols * n_rows + 1):
                zone_count = len(team_df[team_df["zone"] == zone])
                record[f"zone_{zone}_pct"] = zone_count / total * 100 if total > 0 else 0

            # Pas başarı oranı (genel)
            passes = team_df[team_df["type"] == "Pass"]
            if len(passes) > 0 and "pass_outcome" in passes.columns:
                record["pass_success_rate"] = passes["pass_outcome"].isna().mean() * 100
            else:
                record["pass_success_rate"] = 0

            # Şut sayısı
            record["shot_count"] = len(team_df[team_df["type"] == "Shot"])

            # Baskı sayısı
            record["pressure_count"] = len(team_df[team_df["type"] == "Pressure"])

            # Target: Maç sonucu
            if home_score is not None and away_score is not None:
                if team == home_team:
                    if home_score > away_score:
                        record["result"] = "W"
                    elif home_score < away_score:
                        record["result"] = "L"
                    else:
                        record["result"] = "D"
                elif team == away_team:
                    if away_score > home_score:
                        record["result"] = "W"
                    elif away_score < home_score:
                        record["result"] = "L"
                    else:
                        record["result"] = "D"
                else:
                    record["result"] = None
            else:
                record["result"] = None

            records.append(record)

    return pd.DataFrame(records)


def train_predictive_model(
    match_features: pd.DataFrame,
    save_path: str = None,
) -> dict:
    """
    Zone paterninden maç sonucunu tahmin eden model eğitir.

    İki model dener:
    1. Random Forest
    2. XGBoost

    MLOps öğrenme noktaları:
    - Cross-validation
    - Model karşılaştırma
    - Confusion matrix
    - Feature importance
    """
    # Veri hazırlığı
    df = match_features.dropna(subset=["result"]).copy()

    feature_cols = [c for c in df.columns if c not in ["match_id", "team", "result"]]
    X = df[feature_cols].fillna(0)
    y_raw = df["result"]

    # XGBoost string label kabul etmez → LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)  # W=2, L=1, D=0 (alfabetik)
    label_names = le.classes_    # ['D', 'L', 'W']

    print(f"\n   📊 Veri seti: {len(X)} örnek, {len(feature_cols)} özellik")
    print(f"   Sınıf dağılımı: {dict(zip(label_names, [sum(y==i) for i in range(len(label_names))]))}")

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Model 1: Random Forest ──
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y))), shuffle=True, random_state=42)
    rf_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")
    rf.fit(X_scaled, y)

    print(f"\n   🌲 Random Forest:")
    print(f"      CV Accuracy: {rf_scores.mean():.3f} (±{rf_scores.std():.3f})")

    # ── Model 2: XGBoost ──
    xgb = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=42, eval_metric="mlogloss",
    )
    xgb_scores = cross_val_score(xgb, X_scaled, y, cv=cv, scoring="accuracy")
    xgb.fit(X_scaled, y)

    print(f"\n   🚀 XGBoost:")
    print(f"      CV Accuracy: {xgb_scores.mean():.3f} (±{xgb_scores.std():.3f})")

    # En iyi modeli seç
    best_model_name = "Random Forest" if rf_scores.mean() >= xgb_scores.mean() else "XGBoost"
    best_model = rf if rf_scores.mean() >= xgb_scores.mean() else xgb
    best_scores = rf_scores if rf_scores.mean() >= xgb_scores.mean() else xgb_scores

    print(f"\n   🏆 En iyi model: {best_model_name} (Accuracy: {best_scores.mean():.3f})")

    # ── Görselleştirme: Feature Importance ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0d1117")

    # Feature Importance
    importances = best_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:10]  # Top 10

    ax1.barh(
        range(len(sorted_idx)),
        importances[sorted_idx][::-1],
        color="#00aaff",
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.set_yticks(range(len(sorted_idx)))
    ax1.set_yticklabels([feature_cols[i] for i in sorted_idx][::-1], fontsize=10, color="white")
    ax1.set_xlabel("Importance", fontsize=12, color="white")
    ax1.set_title(f"Feature Importance ({best_model_name})", fontsize=13, fontweight="bold", color="white")
    ax1.set_facecolor("#1a1a2e")
    ax1.tick_params(colors="white")

    # Model karşılaştırma
    models = ["Random Forest", "XGBoost"]
    means = [rf_scores.mean(), xgb_scores.mean()]
    stds = [rf_scores.std(), xgb_scores.std()]
    colors = ["#00aaff", "#ff6b6b"]

    bars = ax2.bar(models, means, yerr=stds, color=colors, alpha=0.8,
                   edgecolor="white", linewidth=0.5, capsize=5)
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{mean:.3f}", ha="center", va="bottom", fontsize=14,
                 fontweight="bold", color="white")

    ax2.set_ylabel("CV Accuracy", fontsize=12, color="white")
    ax2.set_title("Model Karşılaştırması", fontsize=13, fontweight="bold", color="white")
    ax2.set_facecolor("#1a1a2e")
    ax2.tick_params(colors="white")
    ax2.set_ylim(0, 1.0)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"   📸 Kaydedildi: {save_path}")

    plt.close(fig)

    # ── Classification Report ──
    y_pred = best_model.predict(X_scaled)
    print(f"\n   📋 Classification Report ({best_model_name}):")
    print(classification_report(y, y_pred, target_names=label_names, zero_division=0))

    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "rf_model": rf,
        "xgb_model": xgb,
        "rf_cv_mean": rf_scores.mean(),
        "rf_cv_std": rf_scores.std(),
        "xgb_cv_mean": xgb_scores.mean(),
        "xgb_cv_std": xgb_scores.std(),
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


# ═══════════════════════════════════════════════
#  PART 3: MLFLOW INTEGRATION
# ═══════════════════════════════════════════════

def run_mlflow_experiment(
    clustering_results: dict,
    prediction_results: dict,
    team_features: pd.DataFrame,
    experiment_name: str = "sports-zone-analytics",
):
    """
    MLflow ile tüm deneyleri loglar.

    MLOps öğrenme noktaları:
    - Experiment tracking
    - Parameter & metric logging
    - Artifact logging (görseller, modeller)
    - Model registry
    """
    if not HAS_MLFLOW:
        print("   ⚠️ MLflow yüklü değil, loglama atlanıyor.")
        return

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    # ── Run 1: Clustering ──
    with mlflow.start_run(run_name="clustering_kmeans"):
        # Parametreler
        mlflow.log_param("algorithm", "KMeans")
        mlflow.log_param("n_clusters", clustering_results["n_clusters"])
        mlflow.log_param("n_teams", len(team_features))
        mlflow.log_param("n_features", team_features.shape[1])

        # Metrikler
        mlflow.log_metric("silhouette_score", clustering_results["silhouette_score"])

        # Artifact'ler (görseller)
        for fig_file in ["cluster_profiles.png", "optimal_k.png"]:
            fig_path = f"outputs/figures/{fig_file}"
            if os.path.exists(fig_path):
                mlflow.log_artifact(fig_path, "figures")

        # Model
        mlflow.sklearn.log_model(clustering_results["model"], "kmeans_model")

        print("   ✅ MLflow Run: Clustering logged")

    # ── Run 2: Predictive Model ──
    with mlflow.start_run(run_name="prediction_match_result"):
        # Parametreler
        mlflow.log_param("best_model", prediction_results["best_model_name"])
        mlflow.log_param("n_features", len(prediction_results["feature_cols"]))

        # Metrikler
        mlflow.log_metric("rf_cv_accuracy_mean", prediction_results["rf_cv_mean"])
        mlflow.log_metric("rf_cv_accuracy_std", prediction_results["rf_cv_std"])
        mlflow.log_metric("xgb_cv_accuracy_mean", prediction_results["xgb_cv_mean"])
        mlflow.log_metric("xgb_cv_accuracy_std", prediction_results["xgb_cv_std"])

        # Artifact'ler
        fig_path = "outputs/figures/model_comparison.png"
        if os.path.exists(fig_path):
            mlflow.log_artifact(fig_path, "figures")

        # En iyi modeli kaydet (trusted types gerekli — tree-based modeller için)
        sklearn_trusted = ["sklearn.tree._tree.Tree"]
        if prediction_results["best_model_name"] == "Random Forest":
            mlflow.sklearn.log_model(
                prediction_results["best_model"], "best_model",
                skops_trusted_types=sklearn_trusted,
            )
        else:
            mlflow.xgboost.log_model(prediction_results["best_model"], "best_model")

        print("   ✅ MLflow Run: Prediction logged")

    print(f"\n   🔗 MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
