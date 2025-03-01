import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
from sqlalchemy import create_engine
st.set_page_config(page_title="Poker Analysis", layout='wide', initial_sidebar_state="collapsed", page_icon="🃏")


def style_plotly_figure(fig: Figure) -> Figure:
    fig.update_traces(
        textfont_size=16,
        marker_color='#8C0F31'
    )
    fig.update_layout(
        font=dict(size=18),
        xaxis=dict(
            tickfont_size=12,
            title_font_size=18
        ),
        yaxis=dict(
            tickfont_size=18,
            title_font_size=18
        ),
        margin=dict(l=100, r=100, t=0, b=80),
        height=650,
    )
    return fig


# Loading all data TODO Add caching
df_ten_biggest_pots = pd.read_csv("dashboard_data/ten_biggest_pots.csv")
df_avgwinnings = pd.read_csv("dashboard_data/avg_winnings.csv")
df_ten_most_profitable = pd.read_csv("dashboard_data/ten_most_profitable.csv")

# Sidebar
st.sidebar.markdown("""
### **Positions**

| **Position** | **Description** |
|-------------|------------------|
| **SB (Small Blind)** | Forced bet before cards are dealt. |
| **BB (Big Blind)** | Larger forced bet before cards are dealt. |
| **UTG (Under the Gun)** | First player to act preflop (early position). |
| **UTG+1, UTG+2, UTG+3** | Early positions at a full table (9-max). |
| **MP (Middle Position)** | Plays after UTG but before late positions. |
| **CO (Cutoff)** | One seat before the Button—often raises or steals blinds. |
| **BTN (Button)** | Best position—acts last post-flop, ideal for bluffing. |

---

### **Key Poker Statistics**

| **Stat** | **Definition** |
|----------|---------------|
| **VPIP (Voluntarily Put Money in Pot)** | % of hands where a player **calls or raises preflop** (indicates looseness). |
| **PFR (Preflop Raise %)** | % of hands where a player **raises preflop** (shows aggression). |
| **AF (Aggression Factor)** | (Bet + Raise) / Call (high AF = aggressive player). |
| **3-Bet %** | How often a player **reraises** before the flop (indicates preflop aggression). |
| **C-Bet (Continuation Bet %)** | How often a player **bets the flop** after raising preflop. |
| **WSD (Went to Showdown %)** | How often a player reaches showdown when they see the river. |
""")


# Main Title
st.title("Poker Analysis Project")
st.markdown(
    """
    This dashboard presents an analysis of over 20 million poker hands played since 2009.
    For a quick reference on poker notation, please open the sidebar (click the arrow at the top left).
    Explore the complete project on [GitHub](https://github.com/Morozov173/poker-data-analysis-).
    """
)

# Analysis by categories
tabs = st.tabs(["**General Analysis**", "**Positional Analysis**",
               "**Trends Across Stake Levels**"])


# General Analysis
with tabs[0]:
    st.header("General Analysis")

    # 10 Biggest Pots Played
    with st.container(border=True):
        header_placeholder = st.empty()
        header_placeholder.subheader(
            f"10 Biggest Pots Played in VARIANT-NL - SEAT_COUNT-Max")

        # Slider & Dropdown for data selection
        columns = st.columns([0.1, 1, 0.1, 1, 0.1])
        with columns[1]:
            variant = st.select_slider(
                'Stake Level', [25, 50, 100, 200, 400, 600, 1000], value=1000)
        with columns[3]:
            seat_count = st.selectbox("Table Type (Seat Count)", [6, 9])

        header_placeholder.subheader(
            f"Biggest Pots Played in {variant}NL - {seat_count}Max.")

        df_biggest_pots_filtered = df_ten_biggest_pots[(df_ten_biggest_pots['variant'] == variant) & (
            df_ten_biggest_pots['seat_count'] == seat_count)]

        fig = px.bar(
            df_biggest_pots_filtered,
            x="game_id",
            y="final_pot",
            text="final_pot",
            labels={"game_id": "Game ID", "final_pot": "Pot Size ($)"}
        )
        fig = style_plotly_figure(fig)
        fig.update_traces(
            texttemplate="$%{y:,.0f}",
            textposition="outside",
            textfont_size=16,
        )
        fig.update_layout(
            xaxis_type="category"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Wow. So much meaningful and insightful commnentary in this line of text. Simply Amazing")

        with st.expander("SQL Query Used"):
            query = """
                SELECT
                    pg.position,
                    g.seat_count,
                    g.variant,
                    COUNT(*) AS hands_played,
                    SUM(pg.winnings) AS total_winnings,
                    AVG(pg.winnings) AS avg_winnings_per_hand
                FROM players_games AS pg
                INNER JOIN game_types_info AS g ON pg.game_id = g.game_id
                GROUP BY pg.position, g.seat_count, g.variant;
            """
            st.code(query, "sql")

    # 10 Most Profitable Players
    with st.container(border=True):
        header_placeholder = st.empty()
        header_placeholder.subheader(
            f"10 Most Profitable Players in VARIANT-NL - SEAT_COUNT-Max.")

        # Stake Level & Seat_Count Selection by user
        columns = st.columns([0.1, 1, 0.1, 1, 0.1])
        with columns[1]:
            variant = st.select_slider('Stake Level', [25, 50, 100, 200, 400, 600, 1000], value=1000, key="ten_most_profitable_slider")
        with columns[3]:
            seat_count = st.selectbox("Table Type (Seat Count)", [6, 9], key="ten_most_profitable_selectbox")

        header_placeholder.subheader(f"10 Most Profitable Players in {variant}NL - {seat_count}Max.")

        df_ten_most_profitable_filtered = df_ten_most_profitable[(df_ten_most_profitable['variant'] == variant) & (df_ten_most_profitable['seat_count'] == seat_count)]
        df_ten_most_profitable_filtered.sort_values("amount_won", ascending=True, inplace=True)
        fig = px.bar(
            df_ten_most_profitable_filtered,
            x="player_id",
            y="amount_won",
            text="amount_won",
            labels={"player_id": "Player ID", "amount_won": "$ Won"}
        )
        fig = style_plotly_figure(fig)
        fig.update_traces(
            texttemplate="$%{y:,.0f}",
            textposition="outside",
        )
        fig.update_xaxes(tickfont_size=10)

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Wow. So much meaningful and insightful commnentary in this line of text. Simply Amazing")

        with st.expander("SQL Query Used"):
            query = """
                WITH player_winnings_by_type AS (
                    SELECT 
                        pg.player_id,
                        gti.variant,
                        gti.seat_count,
                        SUM(winnings) AS amount_won,
                        COUNT(*) AS hands_played
                    FROM players_games AS pg
                    JOIN game_types_info AS gti
                    ON pg.game_id = gti.game_id
                    GROUP BY pg.player_id, gti.variant, gti.seat_count
                    ORDER BY gti.variant, gti.seat_count ASC, amount_won DESC
                )
                SELECT *
                FROM (
                    SELECT 
                        *,
                        ROW_NUMBER() OVER (PARTITION BY variant, seat_count) AS rn
                    FROM player_winnings_by_type
                )
                WHERE rn <= 10;
            """
            st.code(query, "sql")

# ------------------------------------------------------------------------------------------------------------


st.divider()

st.subheader("Most profitable players")
st.caption("Some very insightful commentary")

with tabs[1]:
    st.header("Positional Analysis")
    st.subheader("Average Winnings per Hand played by Position")

    if seat_count == 6:
        position_order = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
        custom_labels = {
            "p1": "SB",
            "p2": "BB",
            "p3": "UTG",
            "p4": "MP",
            "p5": "CO",
            "p6": "BTN"
        }
    else:
        position_order = ["UTG", "UTG+1", "UTG+2",
                          "UTG+3", "MP", "CO", "BTN", "SB", "BB"]
        custom_labels = {
            "p1": "SB",
            "p2": "BB",
            "p3": "UTG",
            "p4": "UTG+1",
            "p5": "UTG+2",
            "p6": "UTG+3",
            "p7": "MP",
            "p8": "CO",
            "p9": "BTN"
        }

    df_avgwinnings_filtered = df_avgwinnings[(df_avgwinnings['seat_count'] == seat_count) & (
        df_avgwinnings['variant'] == variant)].copy()
    df_avgwinnings_filtered['position'] = df_avgwinnings_filtered['position'].replace(
        custom_labels)
    df_avgwinnings_filtered['position'] = pd.Categorical(
        df_avgwinnings_filtered['position'], categories=position_order, ordered=True)
    df_avgwinnings_filtered = df_avgwinnings_filtered.sort_values(
        by="position")

    colors = ['#D91147'] * len(df_avgwinnings_filtered)
    colors[-1] = '#8C0F31'
    colors[-2] = '#8C0F31'

    fig = px.bar(
        df_avgwinnings_filtered,
        x='position',
        y='avg_winnings_per_hand',
        text='avg_winnings_per_hand',
        labels={"avg_winnings_per_hand": "Avg. $ Won/Hand",
                "position": "Position"}
    )
    fig.update_traces(
        texttemplate="%{y:.2f}$",
        textposition="outside",
        textfont_size=20,
        marker_color=colors
    )
    fig.update_layout(
        font=dict(size=18),
        xaxis=dict(
            tickfont_size=12,        # X-axis tick label size
            title_font_size=18       # X-axis title size
        ),
        yaxis=dict(
            tickfont_size=18,        # Y-axis tick label size
            title_font_size=18       # Y-axis title size
        ),
        margin=dict(l=60, r=60, t=0, b=50),
        height=600,
        width=1200,
    )
    st.plotly_chart(fig)

    st.subheader("VPIP & PFR Percentages by Position")
    st.subheader("3Bet Percetnages by Position")
