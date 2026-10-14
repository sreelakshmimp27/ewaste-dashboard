import pandas as pd
import streamlit as st

st.set_page_config(page_title="E-Waste Hotspot Dashboard", layout="wide")

DATA = "data/"  # put all CSV files in a folder named 'data' next to this file


@st.cache_data
def load():
    return dict(
        total=pd.read_csv(DATA + "district_total.csv"),
        monthly=pd.read_csv(DATA + "monthly.csv"),
        quarter=pd.read_csv(DATA + "district_quarter.csv"),
        category=pd.read_csv(DATA + "district_category.csv"),
        coords=pd.read_csv(DATA + "district_coords.csv"),
        hot=pd.read_csv(DATA + "hotspots.csv"),
    )


d = load()
hot = d["hot"].merge(d["coords"], on="district")

st.title("E-Waste Generation and Recycling Hotspots - Kerala")
st.caption("Big Data pipeline: HDFS > Pig > Hive > Spark MLlib > Streamlit. "
           "Data is simulated, modelled on CPCB/KSPCB patterns.")

# ---- Key stats ----
top = d["total"].sort_values("total_qty", ascending=False).iloc[0]
peak = d["monthly"].sort_values("total_qty", ascending=False).iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total e-waste (units)", f"{int(d['total']['total_qty'].sum()):,}")
c2.metric("Top district", top["district"])
c3.metric("Peak month", peak["month"])
c4.metric("High-zone districts", int((hot["zone"] == "High").sum()))

# ---- Hotspot map and table ----
st.subheader("Predicted hotspots for next quarter")
left, right = st.columns([3, 2])

colors = {"High": [220, 38, 38, 200], "Medium": [245, 158, 11, 200], "Low": [34, 197, 94, 200]}
map_df = hot.rename(columns={"lat": "latitude", "lon": "longitude"}).copy()
map_df["color"] = map_df["zone"].map(colors)
map_df["size"] = map_df["pred_next_qtr"] * 0.4
with left:
    try:
        import pydeck as pdk
        layer = pdk.Layer("ScatterplotLayer", map_df, get_position="[longitude, latitude]",
                          get_fill_color="color", get_radius="size",
                          radius_min_pixels=6, radius_max_pixels=26,
                          pickable=True, opacity=0.85)
        view = pdk.ViewState(latitude=10.3, longitude=76.3, zoom=6.3)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                                 tooltip={"text": "{district}\n{zone} zone\n{pred_next_qtr}"}))
    except Exception:
        st.map(map_df[["latitude", "longitude"]])


def colour_zone(v):
    return {
        "High": "background-color:#7f1d1d; color:#fecaca; font-weight:600",
        "Medium": "background-color:#78350f; color:#fde68a; font-weight:600",
        "Low": "background-color:#14532d; color:#bbf7d0; font-weight:600",
    }.get(v, "")


with right:
    show = hot[["district", "pred_next_qtr", "zone"]].sort_values("pred_next_qtr", ascending=False)
    show["pred_next_qtr"] = show["pred_next_qtr"].round(0).astype(int)
    st.dataframe(show.style.map(colour_zone, subset=["zone"]), hide_index=True, use_container_width=True)

st.subheader("Priority sites for new collection centres")
prio = hot[hot["zone"] == "High"].sort_values("pred_next_qtr", ascending=False)
st.write(", ".join(prio["district"]) if len(prio) else "No High-zone districts.")

# ---- Trends ----
st.subheader("Trends")
a, b = st.columns(2)
with a:
    st.markdown("**Total e-waste by district**")
    st.bar_chart(d["total"].set_index("district")["total_qty"].sort_values(ascending=False))
with b:
    st.markdown("**Monthly e-waste (festival-season spikes)**")
    st.line_chart(d["monthly"].set_index("month")["total_qty"])

st.markdown("**Quarterly trend for a selected district**")
pick = st.selectbox("District", sorted(d["quarter"]["district"].unique()))
q = d["quarter"][d["quarter"]["district"] == pick].sort_values("year_quarter")
st.line_chart(q.set_index("year_quarter")["total_qty"])

st.markdown("**Category mix by district**")
pivot = d["category"].pivot(index="district", columns="category", values="total_qty")
st.bar_chart(pivot)
