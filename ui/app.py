"""Streamlit UI. A thin client: every number comes from the FastAPI service."""

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
FRONTIER = "#2a78d6"  # categorical slot 1 (blue)
CURRENT = "#eb6834"   # categorical slot 2 (orange)

EXAMPLES = [
    "Allocate this week's orders.",
    "What does a 10% carbon cut cost?",
    "What is the cheapest plan that is on time?",
    "Plant B loses 30% capacity from 2026-11-02 for two weeks. Re-plan.",
    "Why was order 77 built in Plant C?",
    "Write the brief for the leadership team.",
]

st.set_page_config(page_title="Order Allocation", layout="wide")


def call(method: str, path: str, **kw):
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    try:
        r = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=180, **kw)
    except requests.RequestException as e:
        st.error(f"API not reachable at {API_URL}: {e}")
        return None
    if r.status_code >= 400:
        st.error(f"API error {r.status_code}: {r.text[:500]}")
        return None
    return r.json()


def money(x: float) -> str:
    return f"INR {x / 1e6:,.3f}M" if abs(x) >= 1e6 else f"INR {x:,.0f}"


# ---------- sidebar ----------

with st.sidebar:
    st.header("Settings")
    health = call("get", "/health")
    if health:
        ds = health["data_store"]
        st.caption(f"API {health['api']} · data v{ds.get('data_version')} "
                   f"({'valid' if ds.get('data_valid') else 'INVALID'}) · "
                   f"LLM {health['llm']['mode']} ({health['llm']['model']})")
    carbon_w = st.slider("Carbon weight (cost = 1 − carbon)", 0.0, 1.0, 0.2, 0.05,
                         help="Default 0.2: the knee of the frontier, most of the achievable "
                              "carbon cut for a small cost increase. The response shows the "
                              "implied carbon price per tonne.")
    lateness_w = st.slider("Lateness weight", 0.0, 5.0, 1.0, 0.5,
                           help="1 = minimise late orders first, then cost and carbon. "
                                "Lower values let cost/carbon trade against lateness.")
    apply_events = st.checkbox("Apply loaded capacity events", value=True)
    stability = st.checkbox("Keep re-plans stable (penalise moving orders)", value=True)

weights = {"cost": round(1 - carbon_w, 4), "carbon": carbon_w, "lateness": lateness_w}


def analyze(text: str, scenario: dict | None = None):
    sc = {"apply_loaded_events": apply_events, "stability": stability, **(scenario or {})}
    with st.spinner("Solving…"):
        res = call("post", "/analyze", json={"request": text, "weights": weights,
                                             "scenario": sc})
    if res:
        st.session_state["result"] = res


# ---------- rendering ----------

def tradeoff_chart(rows: list[dict], current: dict | None):
    pts = [r for r in rows if r.get("carbon_weight") is not None]
    if not pts:
        return
    df = pd.DataFrame(pts).sort_values("carbon_t")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["carbon_t"], y=df["cost"], mode="lines+markers+text", name="Optimal plans",
        line={"color": FRONTIER, "width": 2}, marker={"size": 10, "color": FRONTIER},
        text=[f"{n} late" for n in df["late_orders"]], textposition="top center",
        customdata=df[["carbon_weight", "late_orders", "carbon_cut_pct"]],
        hovertemplate="Carbon %{x:,.1f} t<br>Cost INR %{y:,.0f}<br>Carbon weight "
                      "%{customdata[0]}<br>Late orders %{customdata[1]}<br>Carbon cut "
                      "%{customdata[2]}%<extra></extra>"))
    if current:
        fig.add_trace(go.Scatter(
            x=[current["total_carbon_t"]], y=[current["total_cost"]], mode="markers",
            name="This plan", marker={"size": 14, "color": CURRENT, "symbol": "diamond",
                                      "line": {"width": 2, "color": "white"}},
            hovertemplate="This plan<br>Carbon %{x:,.1f} t<br>Cost INR %{y:,.0f}<br>"
                          f"Late orders {current['late_orders']}<extra></extra>"))
    fig.update_layout(
        title="Cost vs carbon of optimal plans (labels: late orders)",
        xaxis_title="Total carbon (t CO2e)", yaxis_title="Total landed cost (INR)",
        height=420, margin={"t": 60, "b": 40}, legend={"orientation": "h", "y": -0.2},
        hovermode="closest")
    st.plotly_chart(fig, width="stretch")
    with st.expander("Trade-off table"):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render(res: dict):
    kind = res["answer_type"]
    badge = {"allocation": "🟢", "scenario": "🔵", "brief": "📝", "infeasible": "🟠",
             "clarification_needed": "❓", "refused": "⛔"}.get(kind, "")
    st.subheader(f"{badge} {kind.replace('_', ' ')}")
    st.markdown(res.get("narrative") or res["summary"])

    k = res.get("kpis") or {}
    if k and res.get("explanation"):
        st.caption("Totals below are for the whole plan, not just this order.")
    if k:
        c = st.columns(5)
        c[0].metric("Total cost", money(k["total_cost"]))
        c[1].metric("Carbon", f"{k['total_carbon_t']:,.1f} t")
        c[2].metric("Late orders", k["late_orders"])
        c[3].metric("Cannot place", k.get("unassigned_orders", 0))
        c[4].metric("Utilisation", " / ".join(f"{p} {u:.0%}"
                                              for p, u in k["plant_utilisation"].items()))
    v = res.get("validator") or {}
    if v:
        st.caption("Validator: " + " · ".join(
            f"{name.replace('_ok', '')} {'✓' if ok else '✗'}" for name, ok in v.items()
            if name != "valid"))

    sc = res.get("scenario_comparison")
    if sc:
        c = st.columns(4)
        c[0].metric("Extra cost", money(sc["extra_cost"]), f"{sc['extra_cost_pct']:+.2f}%",
                    delta_color="inverse")
        c[1].metric("Extra carbon", f"{sc['extra_carbon_t']:+,.1f} t", delta_color="off")
        c[2].metric("Extra late orders", f"{sc['extra_late_orders']:+d}")
        c[3].metric("Orders moved", sc["orders_moved"],
                    f"{sc['plan_stability_pct']:.0f}% unchanged", delta_color="off")
        if sc["capacity_changes"]:
            st.markdown("**Capacity changes**")
            st.dataframe(pd.DataFrame(sc["capacity_changes"]), hide_index=True)
        if sc["moved"]:
            st.markdown("**What moved**")
            st.dataframe(pd.DataFrame(sc["moved"]), width="stretch", hide_index=True)

    if res.get("tradeoff"):
        tradeoff_chart(res["tradeoff"], k or None)

    if res.get("infeasible_orders"):
        st.markdown("**Orders that cannot be on time (closest option)**")
        st.dataframe(pd.DataFrame(res["infeasible_orders"]), hide_index=True)

    if res.get("recommendations"):
        st.markdown("**Recommendations**")
        for r in res["recommendations"]:
            flag = " · needs approval" if r["requires_approval"] else ""
            st.markdown(f"- **{r['action']}** — {r['rationale']} "
                        f"(impact: {r['estimated_impact']}; confidence {r['confidence']}{flag})")

    if res.get("allocation"):
        with st.expander(f"Allocation ({len(res['allocation'])} orders)",
                         expanded=kind == "allocation" and len(res["allocation"]) <= 25):
            df = pd.DataFrame(res["allocation"])
            st.dataframe(df.style.apply(
                lambda r: ["background-color: rgba(235,104,52,0.18)" if not r["on_time"]
                           else "" for _ in r], axis=1),
                width="stretch", hide_index=True)
            st.download_button("Download CSV", df.to_csv(index=False), "allocation.csv")

    with st.expander("Assumptions and limitations"):
        for a in res.get("assumptions", []):
            st.markdown(f"- {a}")
        st.markdown("**Limitations**")
        for lim in res.get("limitations", []):
            st.markdown(f"- {lim}")
    with st.expander(f"Trace {res['run_id']}"):
        trace = call("get", f"/trace/{res['run_id']}")
        if trace:
            st.caption(f"{trace.get('llm_calls', 0)} LLM calls · {trace.get('tokens', 0)} "
                       f"tokens · {trace.get('latency_ms', 0):,.0f} ms · intent from "
                       f"{trace.get('intent_source', '-')}")
            st.json(trace["steps"], expanded=False)


# ---------- pages ----------

st.title("Multi-plant order allocation")
ask, whatif, data = st.tabs(["Ask", "What-if", "Data"])

with ask:
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, width="stretch"):
            st.session_state["q"] = ex
            analyze(ex)
    q = st.text_input("Ask a question or request a plan", key="q")
    if st.button("Run", type="primary") and q.strip():
        analyze(q)

with whatif:
    st.markdown("Run changed inputs through the solver and compare with the base plan.")
    with st.form("cap"):
        c = st.columns(4)
        plant = c[0].selectbox("Plant", ["A", "B", "C"])
        loss = c[1].number_input("Capacity lost (%)", 1, 100, 30)
        d_from = c[2].date_input("From", value=None)
        d_to = c[3].date_input("To", value=None)
        if st.form_submit_button("Re-plan with capacity loss"):
            change = {"plant": plant, "loss_pct": loss / 100}
            if d_from:
                change["from_date"] = d_from.isoformat()
            if d_to:
                change["to_date"] = d_to.isoformat()
            analyze(f"Plant {plant} loses {loss}% capacity. Re-plan.",
                    {"capacity_changes": [change]})
    with st.form("cut"):
        cut = st.number_input("Carbon cut to price (%)", 0.5, 50.0, 2.0, 0.5)
        if st.form_submit_button("Price the carbon cut"):
            analyze(f"What does a {cut}% carbon cut cost?", {"carbon_cut_pct": cut})
    if st.button("Re-plan for all loaded capacity events"):
        analyze("Re-plan for the loaded capacity events", {"replan_events": True})

with data:
    rep = call("get", "/validation")
    if rep:
        st.markdown(f"**Validation: {'VALID' if rep['valid'] else 'INVALID'}**")
        st.code(rep["report"], language=None)
    with st.form("ingest"):
        table = st.selectbox("Table", ["orders", "plants", "transport_duties",
                                       "capacity_events"])
        mode = st.radio("Mode", ["replace", "append"], horizontal=True)
        up = st.file_uploader("CSV file (Annex B schema)", type=["csv"])
        if st.form_submit_button("Upload") and up is not None:
            res = call("post", "/ingest", files={"file": (up.name, up.getvalue(), "text/csv")},
                       data={"meta": f'{{"table": "{table}", "mode": "{mode}"}}'})
            if res:
                st.json(res)

if "result" in st.session_state:
    st.divider()
    render(st.session_state["result"])
