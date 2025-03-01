import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
from sqlalchemy import create_engine
st.set_page_config(page_title="Poker Analysis", layout='wide', initial_sidebar_state="collapsed", )  # page_icon="🃏"


def style_plotly_figure(fig: Figure) -> Figure:
    fig.update_traces(
        textfont_size=16,
        marker_color="#D1495B"  # "#D1495B"  # '#8C0F31'
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
df_ten_least_profitable = pd.read_csv("dashboard_data/ten_least_profitable.csv")
df_action_type_rates = pd.read_csv("dashboard_data/action_type_rates.csv")
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
    This dashboard presents an analysis of over 20 million poker hands played in 2009.
    For a quick reference on poker notation, please open the sidebar (click the arrow at the top left).
    Each chart or graph includes an expander labeled "SQL Query Used," showing exactly how the data was retrieved from the custom PostgreSQL database.
    Explore the complete project on [GitHub](https://github.com/Morozov173/poker-data-analysis-).
    """
)


# Analysis by categories
tabs = st.tabs(["**General Analysis**", "**Positional Analysis**", "**Trends Across Stake Levels**"])


# General Analysis
with tabs[0]:
    st.header("General Analysis")

    # 10 Biggest Pots Played
    with st.container(border=True):
        header_placeholder = st.empty()
        header_placeholder.subheader(f"Biggest Pots Played")
        st.caption("Top 10 final pot values for stake level & table type")

        # Slider & Dropdown for data selection
        columns = st.columns([0.1, 1, 0.1, 1, 0.1])
        with columns[1]:
            variant = st.select_slider(
                'Stake Level', [25, 50, 100, 200, 400, 600, 1000], value=1000)
        with columns[3]:
            seat_count = st.selectbox("Table Type (Seat Count)", [6, 9])

        df_biggest_pots_filtered = df_ten_biggest_pots[(df_ten_biggest_pots['variant'] == variant) & (df_ten_biggest_pots['seat_count'] == seat_count)]

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

    # 10 Most Un/Profitable Players
    with st.container(border=True):
        # Title
        header_placeholder = st.empty()
        header_placeholder.subheader("Most Profitable Players")
        st.caption("Top 10 most profitable for stake level & table type.")

        # Stake Level & Seat_Count Selection by user
        columns = st.columns([0.1, 1, 0.1, 1, 0.1])
        with columns[1]:
            variant = st.select_slider('Stake Level', [25, 50, 100, 200, 400, 600, 1000], value=1000, key="ten_most_profitable_slider")
        with columns[3]:
            seat_count = st.selectbox("Table Type (Seat Count)", [6, 9], key="ten_most_profitable_selectbox")

        columns = st.columns([1, 0.1, 1])
        # Most profitable
        with columns[0]:
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
            fig.update_layout(margin=dict(l=10, r=10, t=0, b=80))
            fig.update_xaxes(tickfont_size=10)

            st.plotly_chart(fig, use_container_width=True)
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
            # st.caption(f"Wow. So much meaningful and insightful commnentary in this line of text. Simply Amazing")

        # Most Loosing
        with columns[2]:
            df_ten_least_profitable_filtered = df_ten_least_profitable[(df_ten_least_profitable['variant'] == variant) & (df_ten_least_profitable['seat_count'] == seat_count)]
            df_ten_least_profitable_filtered.sort_values("amount_won", ascending=False, inplace=True)
            fig = px.bar(
                df_ten_least_profitable_filtered,
                x="player_id",
                y="amount_won",
                text="amount_won",
                labels={"player_id": "Player ID", "amount_won": "$ Lost"}
            )
            fig = style_plotly_figure(fig)
            fig.update_traces(
                texttemplate="$%{y:,.0f}",
                textposition="outside",
                marker_color="#577399"
            )
            fig.update_xaxes(tickfont_size=8)

            st.plotly_chart(fig, use_container_width=True)

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
                    ORDER BY gti.variant, gti.seat_count, amount_won ASC
                    )
                    SELECT *
                    FROM (
                        SELECT
                            *,
                            ROW_NUMBER() OVER (PARTITION BY variant, seat_count) AS rn
                        FROM player_winnings_by_type
                    )
                    WHERE rn <= 10;"""
                st.code(query, "sql")

            # st.caption(f"Wow. So much meaningful and insightful commnentary in this line of text. Simply Amazing")


# Positional Analysis
with tabs[1]:
    st.header("Positional Analysis")
    st.write("")  # Blank line as a spacer

    # Action type rate heatmap
    with st.container(border=True):
        st.subheader("Action Frequencies by Table Position")
        st.caption("A heatmap showing how often each action (fold, check, call, raise) occurs in different positions at the table.")

        # Slider & Dropdown for data selection
        columns = st.columns([0.1, 1, 0.1, 1, 0.1])
        with columns[1]:
            variant = st.select_slider('Stake Level', [25, 50, 100, 200, 400, 600, 1000], value=1000, key="action_types_rates_sld")
        with columns[3]:
            seat_count = st.selectbox("Table Type (Seat Count)", [6, 9], key="action_types_rates_slctbox")
            # Labeling of Positions
        # Labeling of Positions
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
            position_order = ["UTG", "UTG+1", "UTG+2", "UTG+3", "MP", "CO", "BTN", "SB", "BB"]
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

        df_filtered = df_action_type_rates[(df_action_type_rates['variant'] == variant) & (df_action_type_rates['seat_count'] == seat_count)].copy()
        df_filtered['position'] = df_filtered['position'].replace(custom_labels)
        df_filtered['position'] = pd.Categorical(df_filtered['position'], categories=position_order, ordered=True)
        heatmap_data = df_filtered.set_index('position')[['fold_rate', 'check_rate', 'call_rate', 'raise_rate']]
        heatmap_data = heatmap_data.T
        heatmap_data = heatmap_data.reindex(columns=position_order)
        heatmap_data.rename(
            index={
                "fold_rate": "Fold Rate %",
                "check_rate": "Check Rate %",
                "call_rate": "Call Rate %",
                "raise_rate": "Raise Rate %"},
            inplace=True)

        custom_color_scale = [
            (0.0,  "#f8e2e5"),  # Very light pinkish
            (0.5,  "#D1495B"),  # The main color
            (1.0,  "#7a1322")   # A darker red shade
        ]
        fig = px.imshow(
            heatmap_data,
            text_auto=True,
            color_continuous_scale=custom_color_scale,
            labels=dict(x="Position", y="Action Type", color="Rate (%)")
        )
        fig.update_layout(
            xaxis_title="Action Type",
            yaxis_title="Position",
            font=dict(size=18),
            xaxis=dict(tickfont_size=18, title_font_size=18),
            yaxis=dict(tickfont_size=18, title_font_size=18),
            margin=dict(l=100, r=100, t=50, b=100),
            height=650,
            width=1000
        )

        l, m, r = st.columns([0.5, 1.5, 0.5])
        with m:
            st.plotly_chart(fig, use_container_width=True)

        st.caption("""Earlier positions fold more often but also raise decisively, reflecting a cautious yet aggressive approach when acting first. Meanwhile, the big blind shows the highest check rate, which makes sense given the option to see the flop for free if no one raises. Overall, these patterns illustrate how position strongly influences preflop decisions.""")
        with st.expander("SQL Query Used"):
            query = """
                WITH actions_with_variant AS (
                    SELECT
                        a.game_id,
                        a.position,
                        a.action_type,
                        a.amount,
                        gti.variant,
                        gti.seat_count
                    FROM actions AS a
                    JOIN game_types_info AS gti
                    ON a.game_id = gti.game_id
                ), action_type_counts AS(
                    SELECT 
                        a.position,
                        a.variant,
                        a.seat_count,
                        SUM(CASE WHEN action_type = 'f' THEN 1 ELSE 0 END) AS amount_of_folds,
                        SUM(CASE WHEN action_type = 'cc' AND amount = 0 THEN 1 ELSE 0 END) AS amount_of_checks,
                        SUM(CASE WHEN action_type = 'cc' AND amount > 0 THEN 1 ELSE 0 END) AS amount_of_calls,
                        SUM(CASE WHEN action_type = 'cbr' THEN 1 ELSE 0 END) AS amount_of_raises,
                        SUM(CASE WHEN action_type != 'sm' THEN 1 ELSE 0 END) AS total_actions
                        FROM actions_with_variant AS a
                    GROUP BY a.position, a.variant, a.seat_count
                )
                SELECT 
                    a.position,
                    a.variant,
                    a.seat_count,
                    ROUND(amount_of_folds * 100.0 / total_actions, 2) AS fold_rate,
                    ROUND(amount_of_checks * 100.0 / total_actions, 2) AS check_rate,
                    ROUND(amount_of_calls * 100.0 / total_actions, 2) AS call_rate,
                    ROUND(amount_of_raises * 100.0 / total_actions, 2) AS raise_rate
                FROM action_type_counts AS a
                ORDER BY a.variant, a.seat_count, a.position; """
            st.code(query, "sql")

        # Average Winnings by Position
    with st.container(border=True):
        st.subheader("Average Winnings by Position")
        st.caption("The Average $ Won per Hand Played for every Position across stake level & table type")

        # Slider & Dropdown for data selection
        columns = st.columns([0.1, 1, 0.1, 1, 0.1])
        with columns[1]:
            variant = st.select_slider('Stake Level', [25, 50, 100, 200, 400, 600, 1000], value=1000, key="avg_winnings_sld")
        with columns[3]:
            seat_count = st.selectbox("Table Type (Seat Count)", [6, 9], key="avg_winnings_slctbox")
        # Labeling of Positions
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

        df_avgwinnings_filtered = df_avgwinnings[(df_avgwinnings['seat_count'] == seat_count) & (df_avgwinnings['variant'] == variant)].copy()
        df_avgwinnings_filtered['position'] = df_avgwinnings_filtered['position'].replace(custom_labels)
        df_avgwinnings_filtered['position'] = pd.Categorical(df_avgwinnings_filtered['position'], categories=position_order, ordered=True)
        df_avgwinnings_filtered = df_avgwinnings_filtered.sort_values(by="position")

        colors = ["#D1495B"] * len(df_avgwinnings_filtered)
        colors[-1] = "#577399"
        colors[-2] = "#577399"

        fig = px.bar(
            df_avgwinnings_filtered,
            x='position',
            y='avg_winnings_per_hand',
            text='avg_winnings_per_hand',
            labels={"avg_winnings_per_hand": "Avg. $ Won/Hand", "position": "Position"}
        )
        fig = style_plotly_figure(fig)
        fig.update_traces(
            texttemplate="%{y:.2f}$",
            textposition="outside",
            textfont_size=20,
            marker_color=colors
        )
        fig.update_xaxes(tickfont_size=16)
        st.plotly_chart(fig)
        st.caption(f"Wow. So much meaningful and insightful commnentary in this line of text. Simply Amazing")
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
                GROUP BY pg.position, g.seat_count, g.variant; """
            st.code(query, "sql")
