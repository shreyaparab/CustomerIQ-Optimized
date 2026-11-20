import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.state import require_data, setup  # noqa: E402
from src.segmentation.cluster import evaluate_k_range, fit_segments, suggest_label  # noqa: E402

setup("Customer Segments", "🧩")
df = require_data()

st.write("K-Means groups customers by similarity. Segments are **descriptive**, not guaranteed "
         "customer types, and say nothing causal about purchasing.")

ks = evaluate_k_range(df)
c1, c2 = st.columns(2)
c1.plotly_chart(px.line(ks, x="k", y="inertia", markers=True, title="Inertia (elbow)"),
                use_container_width=True)
c2.plotly_chart(px.line(ks, x="k", y="silhouette", markers=True, title="Silhouette score"),
                use_container_width=True)
st.caption("Use these as aids, not automatic truth. Low silhouette values mean weakly separated groups.")

k = st.slider("Number of segments (K)", 2, 8, int(ks.loc[ks["silhouette"].idxmax(), "k"]))
try:
    seg = fit_segments(df, k)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.metric("Silhouette score", f"{seg.silhouette:.3f}")
plot = df.assign(segment=seg.labels.astype(str), PC1=seg.pca_coords[:, 0], PC2=seg.pca_coords[:, 1])
fig = px.scatter(plot, x="PC1", y="PC2", color="segment", opacity=0.6,
                 title="PCA projection (2D, lossy)")
st.plotly_chart(fig, use_container_width=True)
st.caption(f"PC1+PC2 explain {sum(seg.explained_variance):.1%} of variance. "
           "A projection, not proof that clusters are objectively real.")

st.subheader("Segment profiles")
profile = seg.profile.copy()
profile["suggested_label"] = [suggest_label(profile.loc[i], profile) for i in profile.index]
st.dataframe(profile, use_container_width=True)
st.caption("Suggested labels are drafts for human review. Observed purchase rates are descriptive associations.")
st.download_button("Download segment assignments",
                   df.assign(segment=seg.labels).to_csv(index=False).encode(),
                   "segments.csv", "text/csv")
