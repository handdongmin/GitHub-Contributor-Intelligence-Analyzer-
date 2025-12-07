import datetime as dt
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700&display=swap');
:root {
  --bg: #f6f7fb;
  --card: #ffffff;
  --text: #0f172a;
  --muted: #6b7280;
  --accent: #0ea5e9;
  --accent-strong: #0284c7;
  --pill: #e0f2fe;
}
* { font-family: 'Pretendard', 'Inter', system-ui, -apple-system, sans-serif; }
.main, .block-container { background: var(--bg); }
header { background: transparent; }
.stTabs [role="tablist"] { gap: 1rem; }
.stTabs [role="tab"] {
  padding: 0.5rem 0.75rem;
  border-radius: 999px;
  color: var(--muted);
  border: 1px solid #e5e7eb;
  background: #f8fafc;
}
.stTabs [aria-selected="true"] {
  color: var(--accent-strong);
  border-color: var(--accent-strong);
  background: var(--pill);
}
.stSidebar { background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%); }
.stSidebar * { color: #e5e7eb !important; }
.stSidebar input, .stSidebar .stDateInput input {
  background: #111827 !important;
  border: 1px solid #1f2937 !important;
  color: #e5e7eb !important;
}
.stSidebar button {
  background: linear-gradient(90deg, #0ea5e9 0%, #38bdf8 100%) !important;
  border: none !important;
  color: white !important;
  font-weight: 700;
}
.hero {
  padding: 1.25rem 1.5rem;
  border-radius: 18px;
  background: linear-gradient(120deg, #0ea5e9 0%, #38bdf8 50%, #c7d2fe 100%);
  color: #0b1220;
  box-shadow: 0 14px 40px rgba(14, 165, 233, 0.25);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}
.hero h1 { margin: 0 0 0.35rem 0; font-size: 1.8rem; }
.hero p { margin: 0; color: #0f172a; font-weight: 700; font-size: 1.05rem; }
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  background: var(--pill);
  color: var(--accent-strong);
  font-weight: 600;
  font-size: 0.9rem;
}
.hero-img {
  width: 96px;
  height: 96px;
  object-fit: cover;
  border-radius: 16px;
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.55);
}
.card {
  background: var(--card);
  border-radius: 16px;
  padding: 1rem 1.2rem;
  border: 1px solid #e5e7eb;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}
.metric-card {
  background: var(--card);
  border-radius: 14px;
  padding: 0.9rem 1rem;
  border: 1px solid #e5e7eb;
}
.metric-label { color: var(--muted); font-size: 0.9rem; }
.metric-value { font-size: 1.4rem; font-weight: 700; color: var(--text); }
.empty-state {
  background: #e0f2fe;
  color: #0f172a;
  border: 1px solid #bae6fd;
  padding: 0.9rem 1rem;
  border-radius: 12px;
}
.heatmap-summary {
  font-weight: 700;
  font-size: 1rem;
  color: var(--text);
}
</style>
"""


# Small HTTP helpers so all API calls share timeout/error handling.
def post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API 요청 실패: {url} / {exc}")
        return {}


def get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{API_BASE}{path}"
    try:
        resp = requests.get(url, params=params, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API 요청 실패: {url} / {exc}")
        return {}


st.set_page_config(
    page_title="GitHub Contributor Intelligence Analyzer",
    layout="wide",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
# Keep table cells from truncating long text (e.g., summaries).
st.markdown(
    """
    <style>
    td {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: initial !important;
        word-break: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("GitHub Contributor Intelligence Analyzer")
st.caption(
    "GitHub 커밋·이슈로 활동 분석, 요약, 클러스터링, 토픽 모델링, 생산성 예측을 제공하는 대시보드"
)

# ---------- 사이드바: 리포지토리 선택 ----------

st.sidebar.header("GitHub 설정")

username = st.sidebar.text_input("GitHub 사용자명", value="")
if username and st.sidebar.button("사용자 리포 불러오기"):
    repos = get(f"/fetch/user/{username}/repos")
    if isinstance(repos, list) and repos:
        st.session_state["user_repos"] = repos
        st.session_state["repo_selector"] = 1  # 첫 번째 리포지토리를 기본 선택
    else:
        st.session_state["user_repos"] = []
        st.sidebar.warning("가져올 리포지토리가 없습니다.")
        st.session_state["repo_selector"] = 0

repos_state: List[Dict[str, Any]] = st.session_state.get("user_repos", [])
repo_names = [r.get("name", "") for r in repos_state]
repo_options = ["직접 입력"] + [
    f"{r.get('name', '')} (★{r.get('stargazers_count', 0)}, fork {r.get('forks_count', 0)})"
    for r in repos_state
]
current_selector = st.session_state.get("repo_selector", 0)
if current_selector >= len(repo_options):
    current_selector = 0
    st.session_state["repo_selector"] = 0
selected_idx = st.sidebar.selectbox(
    "저장소 목록에서 선택",
    options=list(range(len(repo_options))),
    format_func=lambda i: repo_options[i],
    index=current_selector,
    key="repo_selector",
)
selected_repo = repos_state[selected_idx - 1] if selected_idx > 0 and repos_state else None
selected_owner = (selected_repo.get("owner") or {}).get("login") if selected_repo else None

manual_owner_default = username or "pallets"
manual_repo_default = repo_names[0] if repo_names else "flask"

if selected_repo:
    owner = selected_owner or manual_owner_default
    repo = selected_repo.get("name", manual_repo_default)
    st.sidebar.caption(f"선택된 저장소: {owner}/{repo}")
else:
    owner = st.sidebar.text_input(
        "소유자(Owner)",
        value=manual_owner_default,
        key="owner_input",
    )
    repo = st.sidebar.text_input(
        "저장소(Repository)",
        value=manual_repo_default,
        key="repo_input",
    )

default_since = dt.date.today() - dt.timedelta(days=90)
default_until = dt.date.today()

since_date = st.sidebar.date_input("시작일", value=default_since)
until_date = st.sidebar.date_input("종료일", value=default_until)

MAX_RANGE_DAYS = 90
range_days = (until_date - since_date).days
if range_days > MAX_RANGE_DAYS:
    capped_since = until_date - dt.timedelta(days=MAX_RANGE_DAYS)
    st.sidebar.warning(f"기간이 {range_days}일입니다. 성능을 위해 최근 {MAX_RANGE_DAYS}일로 제한합니다.")
else:
    capped_since = since_date

analyze_button = st.sidebar.button("분석 실행", type="primary")


def build_repo_payload() -> Dict[str, Any]:
    return {
        "owner": owner,
        "repo": repo,
        "since": f"{capped_since.isoformat()}T00:00:00Z",
        "until": f"{until_date.isoformat()}T23:59:59Z",
    }


tab_overview, tab_heatmap, tab_ml = st.tabs(
    ["개요", "활동 히트맵", "ML 실험실"]
)

if "overview_data" not in st.session_state:
    st.session_state["overview_data"] = None

overview_data: Optional[Dict[str, Any]] = st.session_state.get("overview_data")

if analyze_button:
    with st.spinner("GitHub에서 데이터 가져오는 중..."):
        overview_data = post("/analyze/overview", build_repo_payload())
        st.session_state["overview_data"] = overview_data

# ---------- Overview 탭 ----------

with tab_overview:
    st.markdown(
        """
        <div class="hero">
          <div class="pill">개요</div>
          <h1>Repository Pulse</h1>
          <p>커밋·이슈 흐름을 한눈에 보고, 활동 패턴과 주요 주제를 요약합니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 스냅샷")

    if not overview_data:
        st.markdown(
            '<div class="empty-state">좌측에서 소유자/저장소를 지정하고 <b>분석 실행</b> 버튼을 눌러주세요.</div>',
            unsafe_allow_html=True,
        )
    else:
        counts = overview_data.get("counts", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">커밋 수</div>
                  <div class="metric-value">{counts.get("commits", 0)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">이슈 수</div>
                  <div class="metric-value">{counts.get("issues", 0)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
                <div class="metric-card">
                  <div class="metric-label">기간</div>
                  <div class="metric-value">선택된 범위</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        st.markdown("#### 활동 클러스터 (요약/예시 포함)")
        cluster_data = overview_data.get("clusters", {})
        cluster_list = (
            cluster_data.get("clusters")
            if isinstance(cluster_data, dict)
            else cluster_data
        )
        if cluster_list:
            for cluster in cluster_list:
                with st.expander(
                    f"클러스터 {cluster.get('cluster_id')} (size={cluster.get('size')})"
                ):
                    st.write("**요약**")
                    st.write(cluster.get("summary", ""))
                    st.write("**예시 텍스트 (최대 5개)**")
                    for txt in cluster.get("example_texts", []):
                        st.markdown(f"- {txt}")
        else:
            st.markdown(
                '<div class="empty-state">클러스터 정보가 없습니다.</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        st.markdown("#### 토픽 모델링 (LDA)")
        topics_info = overview_data.get("topics", {})
        topics = topics_info.get("topics", [])

        if topics:
            for t in topics:
                if isinstance(t, dict):
                    topic_id = t.get("topic_id")
                    words = t.get("words", [])
                    label = t.get("label", "")
                    keywords = ", ".join(t.get("keywords") or [])
                    summary = t.get("summary", "")
                    examples = t.get("example_texts", [])
                else:
                    topic_id, words = t
                    label = ", ".join(word for word, _ in words[:3])
                    keywords, summary, examples = "", "", []
                pretty_words = ", ".join(f"{w} ({weight:.2f})" for w, weight in words)
                st.markdown(f"**Topic {topic_id} ({label})**")
                st.markdown(f"- 상위 단어: {pretty_words}")
                if keywords:
                    st.markdown(f"- 키워드: {keywords}")
                if summary:
                    st.markdown(f"- 요약: {summary}")
                if examples:
                    st.markdown("  · 예시:")
                    for ex in examples[:2]:
                        st.markdown(f"    · {ex}")
        else:
            st.write("토픽 정보가 없습니다.")

# ---------- Heatmap ----------

with tab_heatmap:
    st.subheader("활동 히트맵 (일별 커밋 잔디)")

    if not overview_data:
        st.markdown(
            '<div class="empty-state">개요 탭에서 <b>분석 실행</b>을 눌러 먼저 데이터를 가져와 주세요.</div>',
            unsafe_allow_html=True,
        )
    else:
        heatmap_points = overview_data.get("heatmap", [])
        if not heatmap_points:
            st.write("히트맵 데이터가 없습니다.")
        else:
            df = pd.DataFrame(heatmap_points)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")

            start_date = df.index.min()
            end_date = df.index.max()
            start_sunday = start_date - pd.Timedelta(days=(start_date.weekday() + 1) % 7)
            end_saturday = end_date + pd.Timedelta(days=(6 - end_date.weekday()) % 7)

            full_range = pd.date_range(start_sunday, end_saturday, freq="D")
            full_df = df.reindex(full_range, fill_value=0)
            full_df["week"] = ((full_df.index - start_sunday).days // 7).astype(int)
            full_df["dow"] = full_df.index.dayofweek  # Monday=0
            full_df["dow_name"] = full_df["dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
            full_df["is_weekend"] = full_df["dow"].isin([5, 6])
            full_df["month"] = full_df.index.month
            full_df["week_total"] = full_df.groupby("week")["count"].transform("sum")
            full_df["month_total"] = full_df.groupby("month")["count"].transform("sum")

            matrix = full_df.pivot(index="dow", columns="week", values="count").fillna(0)

            week_starts = [start_sunday + pd.Timedelta(days=7 * w) for w in matrix.columns]
            month_labels = []
            last_month = None
            for dt_week in week_starts:
                label = f"{dt_week.month}월" if last_month != dt_week.month else ""
                month_labels.append(label)
                last_month = dt_week.month

            # streak 계산
            counts = full_df["count"]
            active_days = (counts > 0).sum()
            total_commits = int(counts.sum())
            max_day = int(counts.max())
            max_day_date = (
                full_df.index[counts == max_day][0].date().isoformat()
                if max_day > 0
                else "-"
            )
            weekday_totals = full_df.groupby("dow")["count"].sum()
            weekday_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            busiest_weekday = weekday_names.get(int(weekday_totals.idxmax()), "-") if weekday_totals.sum() > 0 else "-"
            max_streak = 0
            curr = 0
            for val in counts:
                if val > 0:
                    curr += 1
                else:
                    max_streak = max(max_streak, curr)
                    curr = 0
            max_streak = max(max_streak, curr)

            # Altair 히트맵
            vis_df = full_df.reset_index().rename(columns={"index": "date", "count": "commits"})
            heat = (
                alt.Chart(vis_df)
                .mark_rect(cornerRadius=3)
                .encode(
                    x=alt.X("week:O", axis=alt.Axis(labels=False, ticks=False, title="")),
                    y=alt.Y("dow_name:N", sort=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], title=""),
                    color=alt.Color(
                        "commits:Q",
                        scale=alt.Scale(
                            type="threshold",
                            domain=[1, 2, 4, 8],
                            range=["#e5f2ff", "#cde1ff", "#8fb4ff", "#3e7bdc", "#173c92"],
                        ),
                        legend=alt.Legend(title="커밋 수"),
                    ),
                    opacity=alt.condition("datum.is_weekend", alt.value(0.85), alt.value(1)),
                    tooltip=[
                        alt.Tooltip("date:T", title="날짜"),
                        alt.Tooltip("commits:Q", title="커밋 수"),
                        alt.Tooltip("week_total:Q", title="해당 주 합계"),
                        alt.Tooltip("month_total:Q", title="해당 월 합계"),
                    ],
                )
                .properties(width=max(360, len(week_starts) * 12), height=240)
            )

            month_df = pd.DataFrame({"week": list(range(len(week_starts))), "month": month_labels})
            month_chart = (
                alt.Chart(month_df)
                .mark_text(baseline="bottom", dy=4, fontSize=10, fontWeight="bold", color="#475569")
                .encode(x=alt.X("week:O", axis=alt.Axis(labels=False, ticks=False, title="")), text="month")
                .properties(width=max(360, len(week_starts) * 12), height=20)
            )

            st.markdown(
                f'<div class="heatmap-summary">요약 · 총 커밋 {total_commits} · 활동일 {active_days}일 · 최대 하루 {max_day}건({max_day_date}) · 최장 연속 {max_streak}일 · 가장 활발한 요일 {busiest_weekday}</div>',
                unsafe_allow_html=True,
            )
            composed = alt.vconcat(month_chart, heat).configure_axis(grid=False)
            st.altair_chart(composed, use_container_width=True)

            with st.expander("원본 데이터 보기"):
                df_view = df.reset_index().rename(columns={"date": "날짜", "count": "커밋 수"})
                st.dataframe(df_view, use_container_width=True)

# ---------- ML Lab 탭 ----------

with tab_ml:
    st.subheader("ML 실험실: 요약 · 클러스터링 · 토픽 · 생산성 예측")
    st.caption("샘플 데이터를 수정하며 각 기능이 정상 동작하는지 빠르게 검증해보세요.")

    st.markdown("### 1) 텍스트 요약 (/ml/summarize)")
    # 한 덩어리 텍스트를 받아 전체 맥락을 압축한다.
    st.write("여러 줄이라도 하나의 텍스트 블록으로 받아 전체 맥락을 요약합니다.")
    text_input = st.text_area(
        "요약할 전체 텍스트를 붙여넣어 주세요.",
        height=220,
        value=(
            "Login bug: fix null token handling, add SSO error banner, and log correlation IDs for support. "
            "Performance: add Redis cache layer for project stats API, batch DB reads, and cap slow queries. "
            "Docs: rewrite setup guide with Docker Compose, offline installation notes, and rollback steps. "
            "Monitoring: add Grafana alerts for error spike/latency, plus SLO dashboard with burn-rate alerts. "
            "Refactor: split monolithic cron into focused jobs (cleanup, billing sync, report export) with feature flags. "
            "Security: rotate API keys automatically, enforce HTTPS redirects, and add dependency vulnerability scan. "
            "UX: redesign onboarding wizard, add keyboard shortcuts, and improve empty states with contextual help."
        ),
    )
    if st.button("요약 실행"):
        text = text_input.strip()
        if not text:
            st.warning("요약할 텍스트를 입력해 주세요.")
        else:
            
            result = post("/ml/summarize", {"text": text})
            summary = result.get("summary")
            if summary:
                st.markdown("**요약 결과**")
                st.write(summary)
            else:
                st.write("요약 결과가 없습니다.")

    st.divider()
    st.markdown("### 2) 텍스트 클러스터링 (/ml/cluster)")
    st.write("비슷한 주제를 자동으로 묶습니다. 자동 k 사용 시 문서 수에 맞춰 k를 찾습니다.")
    cluster_text = st.text_area(
        "클러스터링할 텍스트들 (줄바꿈으로 구분)",
        height=140,
        value=(
            "add user authentication with JWT\n"
            "fix login bug on null token\n"
            "improve UI layout for dashboard\n"
            "update README with deployment steps\n"
            "refactor database layer to use repository pattern\n"
            "optimize query performance for reports\n"
            "add alerting rules to monitoring stack\n"
            "reduce bundle size by code splitting"
        ),
    )
    auto_k = st.checkbox("문서 수 기반 자동 k", value=True)
    k = st.slider("클러스터 개수 k", min_value=2, max_value=8, value=3, step=1, disabled=auto_k)
    if st.button("클러스터링 실행"):
        texts = [line for line in cluster_text.splitlines() if line.strip()]
        payload = {"texts": texts}
        if not auto_k:
            payload["k"] = k
        result = post("/ml/cluster", payload)
        clusters = result.get("clusters") or []
        if clusters:
            k_used = result.get("k_used")
            if k_used:
                st.caption(f"사용된 k = {k_used} (자동/입력 기반)")
            table = []
            for c in clusters:
                keywords = ", ".join(c.get("keywords") or [])
                table.append(
                    {
                        "클러스터": c.get("cluster_id"),
                        "개수": c.get("size"),
                        "키워드": keywords,
                        "요약": c.get("summary", ""),
                    }
                )
            # 텍스트 컬럼을 넓게 잡아 요약/키워드가 끊기지 않게 표시.
            st.dataframe(
                pd.DataFrame(table),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "키워드": st.column_config.TextColumn("키워드", width="medium", max_chars=200),
                    "요약": st.column_config.TextColumn("요약", width="large", max_chars=500),
                },
            )
            st.markdown("**요약 전체 보기**")
            for row in table:
                st.markdown(f"- 클러스터 {row['클러스터']} ({row['개수']}개): {row['요약']}")
            st.markdown("**예시 텍스트**")
            for cluster in clusters:
                st.markdown(f"- 클러스터 {cluster.get('cluster_id')}")
                for txt in cluster.get("example_texts", []):
                    st.markdown(f"    · {txt}")
        else:
            labels = result.get("labels", [])
            for t, label in zip(texts, labels):
                st.markdown(f"- [{label}] {t}")

    st.divider()
    st.markdown("### 3) 토픽 모델링 (/ml/topics)")
    st.write("짧은 문서 모음에서 주요 토픽을 뽑고 라벨/키워드/예시를 보여줍니다.")
    topic_text = st.text_area(
        "토픽을 뽑을 텍스트들",
        height=140,
        value=(
            "add user authentication with OAuth2\n"
            "fix login redirect loop bug\n"
            "improve UI layout for analytics dashboard\n"
            "add dark mode theme and color tokens\n"
            "update documentation for API versioning\n"
            "optimize database query performance for weekly reports\n"
            "add structured logging and request tracing\n"
            "refactor caching layer to use redis\n"
            "write integration tests for payment webhook"
        ),
    )
    auto_topics = st.checkbox("문서 수 기반 자동 토픽 수", value=True)
    num_topics = st.slider("토픽 개수", min_value=2, max_value=10, value=3, disabled=auto_topics)
    if st.button("토픽 모델링 실행"):
        texts = [line for line in topic_text.splitlines() if line.strip()]
        payload = {"texts": texts}
        if not auto_topics:
            payload["num_topics"] = num_topics
        result = post("/ml/topics", payload)
        topics_info = result.get("topics", [])
        if topics_info:
            rows = []
            for t in topics_info:
                if isinstance(t, dict):
                    tid = t.get("topic_id")
                    words = t.get("words", [])
                    label = t.get("label", "")
                    keywords = ", ".join(t.get("keywords") or [])
                    summary = t.get("summary", "")
                    examples = t.get("example_texts", [])
                else:
                    tid, words = t
                    label = ", ".join(word for word, _ in words[:3])
                    keywords, summary, examples = "", "", []
                pretty_words = ", ".join(f"{w} ({weight:.2f})" for w, weight in words)
                rows.append(
                    {
                        "토픽": tid,
                        "라벨": label,
                        "키워드": keywords,
                        "상위 단어": pretty_words,
                        "요약": summary,
                    }
                )
            st.dataframe(pd.DataFrame(rows))
            if topics_info and isinstance(topics_info[0], dict):
                st.markdown("**예시 텍스트 (토픽별)**")
                for t in topics_info:
                    st.markdown(f"- 토픽 {t.get('topic_id')} · {t.get('label')}")
                    for txt in t.get("example_texts", [])[:3]:
                        st.markdown(f"    · {txt}")
        else:
            st.write("토픽 결과가 없습니다.")

    st.divider()
    st.markdown("### 4) 생산성 예측 (/ml/predictivity)")
    st.caption("샘플 데이터로 모델 성능(MAE/R2)과 예측값을 확인합니다.")

    st.write("작은 표를 편집해 성능/예측을 확인하세요. score 열은 정답(실제 생산성)입니다.")
    # Seed sample rows for quick what-if edits.
    default_rows = [
        {"commits": 25, "issues": 3, "additions": 500, "deletions": 100, "active_days": 10, "prs": 2, "score": 3.5},
        {"commits": 12, "issues": 2, "additions": 320, "deletions": 80, "active_days": 7, "prs": 1, "score": 2.4},
        {"commits": 38, "issues": 4, "additions": 850, "deletions": 250, "active_days": 14, "prs": 3, "score": 4.2},
        {"commits": 8, "issues": 1, "additions": 150, "deletions": 40, "active_days": 4, "prs": 0, "score": 1.8},
        {"commits": 18, "issues": 2, "additions": 420, "deletions": 120, "active_days": 9, "prs": 2, "score": 2.9},
        {"commits": 30, "issues": 5, "additions": 620, "deletions": 200, "active_days": 12, "prs": 4, "score": 4.0},
        {"commits": 5, "issues": 1, "additions": 90, "deletions": 30, "active_days": 3, "prs": 0, "score": 1.2},
    ]

    if "prod_df" not in st.session_state:
        st.session_state["prod_df"] = pd.DataFrame(default_rows)

    prod_df = st.data_editor(
        st.session_state["prod_df"],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        height=240,
        column_config={
            "commits": st.column_config.NumberColumn("commits", width="small"),
            "issues": st.column_config.NumberColumn("issues", width="small"),
            "additions": st.column_config.NumberColumn("additions", width="medium"),
            "deletions": st.column_config.NumberColumn("deletions", width="medium"),
            "active_days": st.column_config.NumberColumn("active_days", width="small"),
            "prs": st.column_config.NumberColumn("prs", width="small"),
            "score": st.column_config.NumberColumn("정답 score", width="small"),
        },
        key="prod_editor",
    )
    st.session_state["prod_df"] = prod_df

    if st.button("생산성 예측 실행"):
        clean_df = prod_df.fillna(0)
        scores = clean_df["score"].tolist() if "score" in clean_df else []
        stats = [
            {
                "commits": int(row["commits"]),
                "issues": int(row["issues"]),
                "additions": int(row["additions"]),
                "deletions": int(row["deletions"]),
                "active_days": int(row["active_days"]),
                "prs": int(row.get("prs", 0)),
            }
            for _, row in clean_df.iterrows()
        ]
        result = post("/ml/predictivity", {"stats": stats, "scores": scores})
        st.write("**모델 성능 결과**")
        st.json({"mae": result.get("mae"), "r2": result.get("r2")})

        preds = result.get("predictions")
        if preds:
            pred_df = pd.DataFrame({"예측값": [round(float(p), 4) for p in preds]})
            st.dataframe(pred_df, use_container_width=False, hide_index=True)

        fi = result.get("feature_importance")
        if fi:
            st.write("**Feature Importance (회귀 계수)**")
            fi_df = pd.DataFrame(fi, columns=["feature", "coef"])
            fi_df["coef"] = fi_df["coef"].astype(float).round(4)
            st.dataframe(fi_df, use_container_width=False, hide_index=True)
            fig, ax = plt.subplots(figsize=(7.5, 3.5))
            ax.bar(fi_df["feature"], fi_df["coef"])
            ax.set_xlabel("Feature")
            ax.set_ylabel("Coefficient")
            ax.set_title("Productivity feature importance")
            plt.xticks(rotation=45)
            st.pyplot(fig)
