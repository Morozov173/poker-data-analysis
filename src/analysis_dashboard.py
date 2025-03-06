import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.graph_objects import Figure
from pathlib import Path
st.set_page_config(page_title="Poker Analysis", layout='wide', initial_sidebar_state="collapsed", )  # page_icon="🃏"


# Applies styling to a Plotly figure, adjusting text size, margins, and colors. To ensure consistent formatting across multiple plots
def style_plotly_figure(fig: Figure) -> Figure:
    fig.update_traces(
        textfont_size=16,
        marker_color="#D1495B"  # "#D1495B"  # '#8C0F31'
    )
    fig.update_layout(
        font=dict(size=18),
        xaxis=dict(
            tickfont_size=18,
            title_font_size=18
        ),
        yaxis=dict(
            tickfont_size=18,
            title_font_size=18
        ),
        margin=dict(l=100, r=100, t=0, b=80),
        height=700,
    )
    return fig


# Converts a 4-character string representing a poker hand into a standardized hand group. Example: 'Ac9j' -> 'A9o'
def asign_poker_card_group(hand: str) -> str:
    rank = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}

    # Determine the correct ordering of the hand based on rank (higher card first)
    if hand[0] == hand[2] or rank[hand[0]] > rank[hand[2]]:
        hand_group = hand[0]+hand[2]
    else:
        hand_group = hand[2]+hand[0]

    # Append 's' for suited hands and 'o' for offsuit hands
    if hand[0] != hand[2] and hand[1] == hand[3]:
        hand_group += 's'
    elif hand[0] != hand[2] and hand[1] != hand[3]:
        hand_group += 'o'

    return hand_group


# Generates a 13x13 poker hand matrix containing hand labels and corresponding winnings.
# The output consists of two DataFrames:
#   - df_labels: A matrix where each cell contains the hand group (e.g., 'AKo', 'QQ').
#   - df_winnings: A matrix where each cell contains the total winnings for that hand.
def create_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cards = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    labels = []
    winnings = []
    for i in range(13):
        row = []
        value_row = []
        offsuit = False  # Flag to switch from suited to offsuit notation

        for j in range(13):  # every new row
            if cards[i] == cards[j]:  # Pocket pairs
                offsuit = True
                card_type = cards[i]+cards[j]
                row.append(card_type)
                value_row.append(float(df.loc[df['hand_type'] == card_type, 'winnings'].iloc[0]))
                continue

            # Assign suited (above diagonal) or offsuit (below diagonal)
            if offsuit:
                card_type = cards[i]+cards[j]+'o'
            else:
                card_type = cards[j]+cards[i]+'s'

            row.append(card_type)
            value_row.append(float(df.loc[df['hand_type'] == card_type, 'winnings'].iloc[0]))

        labels.append(row)
        winnings.append(value_row)

    df_labels = pd.DataFrame(labels, columns=cards, index=cards)
    df_winnings = pd.DataFrame(winnings, columns=cards, index=cards)
    return df_labels, df_winnings


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Loads & caches the require data for the visualizations


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


df_ten_biggest_pots = load_data(DATA_DIR / "ten_biggest_pots.csv")
df_avgwinnings = load_data(DATA_DIR / "avg_winnings.csv")
df_ten_most_profitable = load_data(DATA_DIR / "ten_most_profitable.csv")
df_ten_least_profitable = load_data(DATA_DIR / "ten_least_profitable.csv")
df_action_type_rates = load_data(DATA_DIR / "action_type_rates.csv")
df_vpip_pfr_percentages = load_data(DATA_DIR / "vpip_pfr_percentages.csv")
df_winnings_by_hand = load_data(DATA_DIR / "winnings_by_hand.csv")
df_vpip_pfr_3bet_across_stakes = load_data(DATA_DIR / "vpip_pfr_3bet_across_stakes.csv")

# Main App logic:

# Set-Up of app sidebar
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

# Displaying main app title * some info
st.title("Poker Analysis Project")
st.markdown(
    """
    This dashboard presents an analysis of over 20 million poker hands played in 2009.
    For a quick reference on poker notation, please open the sidebar (click the arrow at the top left).
    Each chart or graph includes an expander labeled "SQL Query Used," showing exactly how the data was retrieved from the custom PostgreSQL database.
    Explore the complete project on [GitHub](https://github.com/Morozov173/poker-data-analysis-).
    """
)

# Creating Tabs by analysis categories
tabs = st.tabs(["**General Analysis**", "**Positional Analysis**", "**Trends Across Stake Levels**"])


# TAB 1 - General Analysis
with tabs[0]:
    st.header("General Analysis")

    # Heatmap - Winnings by cards dealt
    with st.container(border=True):
        st.subheader(f"Profitibality by Starting Cards")
        st.caption("Heatmap: Darker Colors = Higher Winnings")

        # Slider & Dropdown for stake & table selection
        columns = st.columns([0.1, 1, 0.1, 1, 0.1])
        with columns[1]:
            variant = st.select_slider('Stake Level', [25, 50, 100, 200, 400, 600, 1000, 'ALL'], value='ALL', key="wnngs_by_hand_sld")
        with columns[3]:
            seat_count = st.selectbox("Table Type (Seat Count)", [6, 9], key="wnngs_by_hand_slctbox")

        # Sorting and processing data for display
        if variant == 'ALL':
            df_winnings_by_hand['hand_type'] = df_winnings_by_hand['hand'].apply(asign_poker_card_group)
            df_winnings_by_hand = df_winnings_by_hand.groupby("hand_type")["winnings"].sum().reset_index()
            df_winnings_by_hand = df_winnings_by_hand.sort_values(by="winnings", ascending=False)
        else:
            df_winnings_by_hand = df_winnings_by_hand[(df_winnings_by_hand['variant'] == variant) & (df_winnings_by_hand['seat_count'] == seat_count)]
            df_winnings_by_hand['hand_type'] = df_winnings_by_hand['hand'].apply(asign_poker_card_group)
            df_winnings_by_hand = df_winnings_by_hand.groupby("hand_type")["winnings"].sum().reset_index()
            df_winnings_by_hand = df_winnings_by_hand.sort_values(by="winnings", ascending=False)
        label_matrix, winnings_matrix = create_matrices(df_winnings_by_hand)

        # Creating heatmap
        custom_colorscale = [
            (0.00, "#F3C7C9"),
            (0.25, "#E8A2A7"),
            (0.50, "#D1495B"),
            (0.75, "#B13749"),
            (1.00, "#8F2F44")
        ]
        zmax_value = np.percentile(winnings_matrix.values, 92)  # heatmap will account 93 percent of data so outliers wont skew heatmap sensetivity
        fig = px.imshow(
            winnings_matrix.values,
            x=winnings_matrix.columns,
            y=winnings_matrix.index,
            zmax=zmax_value,
            labels=dict(color="Winnings ($)"),
            color_continuous_scale=custom_colorscale,
            text_auto=False
        )
        fig.update_traces(
            text=label_matrix,
            texttemplate="%{text}",
            textfont_size=18,
            hovertemplate=("<b>Winnings:</b> %{z:$,.2f}<br>" + "<b>Card Pair:<b> %{text}<extra></extra>"),
        )
        fig.update_layout(
            height=900,
            width=900,
            hoverlabel=dict(bgcolor="#0e1117", font_size=22),
            coloraxis_colorbar=dict(showticklabels=False)
        )
        fig.update_xaxes(type='category', showticklabels=False, showline=False, zeroline=False)
        fig.update_yaxes(type='category', showticklabels=False, showline=False, zeroline=False)

        with st.columns([0.4, 1, 0.2])[1]:
            st.plotly_chart(fig, use_container_width=False)
        st.caption(
            "High-value pairs like AA (two Aces) and KK (two Kings) dominate as the most profitable hands, reinforcing their reputation as premium starting hands. "
            "These hands win consistently due to their high equity preflop and strong showdown value. "
            "Meanwhile, suited and connected hands, such as suited broadways and small pocket pairs, can still generate solid returns under the right conditions—"
            "particularly when played aggressively in position or when they hit strong postflop combinations like straights and flushes. "
            "This highlights how profitability isn't just about raw hand strength but also about strategic factors like position, stack depth, and the ability to extract value from opponents."
        )
        with st.expander("SQL Query Used"):
            query = """
                SELECT
                    gti.variant,
                    gti.seat_count,
                    pg.hand,
                    SUM(winnings) AS winnings
                FROM players_games AS pg
                JOIN game_types_info AS gti
                ON pg.game_id = gti.game_id
                WHERE hand != '????'
                GROUP BY pg.hand, gti.variant ,gti.seat_count
                ORDER BY winnings DESC
            """
            st.code(query, "sql")

    # Barchart - Top 10 Biggest Pots
    with st.container(border=True):
        st.subheader(f"Top 10 Biggest Pots")
        st.caption("Top 10 largest final pot size")

        # Slider & Dropdown for data selection
        columns = st.columns([0.1, 1, 0.1, 1, 0.1])
        with columns[1]:
            variant = st.select_slider(
                'Stake Level', [25, 50, 100, 200, 400, 600, 1000], value=1000)
        with columns[3]:
            seat_count = st.selectbox("Table Type (Seat Count)", [6, 9])
        df_biggest_pots_filtered = df_ten_biggest_pots[(df_ten_biggest_pots['variant'] == variant) & (df_ten_biggest_pots['seat_count'] == seat_count)]

        # Creating & Styling chart
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

        # Insights & SQL used
        st.caption("""This chart highlights the biggest hands played at the selected stake level and table type.
                    The largest pots often indicate high-stakes confrontations, where strong hands, aggressive betting, or deep stacks create massive winnings""")
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
                INNER JOIN game_types_info AS g
                ON pg.game_id = g.game_id
                GROUP BY pg.position, g.seat_count, g.variant;
            """
            st.code(query, "sql")

    # Double Barchart - Top 10 Most & Least Profitable Players
    with st.container(border=True):
        st.subheader("Top 10 Most & Least Profitable Players")
        st.caption("Comparing top winners and biggest losses")

        # Stake Level & Seat_Count Selection by user
        columns = st.columns([0.1, 1, 0.1, 1, 0.1])
        with columns[1]:
            variant = st.select_slider('Stake Level', [25, 50, 100, 200, 400, 600, 1000], value=1000, key="ten_most_profitable_slider")
        with columns[3]:
            seat_count = st.selectbox("Table Type (Seat Count)", [6, 9], key="ten_most_profitable_selectbox")

        columns = st.columns([1, 0.1, 1])
        # Barchart 2 - 10 Most profitable
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

        # Barchart 1 - 10 Least Profitable
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

        st.caption("""This chart highlights the most profitable players (left) and the biggest losers (right) for the selected stake level and table type.
                          The distribution shows that while some players accumulate significant winnings, others consistently lose large amounts—indicating variations in skill, strategy, or risk tolerance.""")

# TAB 2 - Positional Analysis
with tabs[1]:
    st.header("Positional Analysis")

    # StackedBars - Action Type Frequencies by Table Position
    with st.container(border=True):
        st.subheader("Action Type Frequencies by Table Position")
        st.caption("Breakdown of fold, raise, check, and call rates by position")

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

        # Preaparing Data for Chart
        df_long = df_filtered.melt(
            id_vars=["position"],  # These columns stay the same in every row (the identifier, like position)
            value_vars=["fold_rate", "check_rate", "call_rate", "raise_rate"],  # These columns will be "melted" into one
            var_name="action_type",  # New column to hold the names (fold_rate, check_rate, etc.)
            value_name="rate"  # New column to hold the corresponding values
        )
        df_long['position'] = df_long['position'].replace(custom_labels)
        df_long['position'] = pd.Categorical(df_long['position'], categories=position_order, ordered=True)
        df_long.sort_values("position", inplace=True)
        df_long["action_type"] = df_long["action_type"].replace({
            "fold_rate": "Fold Rate",
            "check_rate": "Check Rate",
            "call_rate": "Call Rate",
            "raise_rate": "Raise Rate"},
        )

        # Creating & Styling chart
        custom_colors = ["#D1495b", "#577399", "#74A57F", "#9ECE9A"]
        fig = px.bar(
            df_long,
            x="position",
            y="rate",
            color="action_type",
            text="rate",
            labels={"position": "Position", "rate": "Rate (%)", "action_type": "Action Type"},
            barmode="stack",

            color_discrete_sequence=custom_colors
        )
        fig.update_traces(texttemplate="%{y:.0f}%")
        fig.update_layout(
            font=dict(size=18),
            xaxis=dict(tickfont_size=18, title_font_size=18),
            yaxis=dict(tickfont_size=18, title_font_size=18),
            margin=dict(l=100, r=100, t=0, b=80),
            height=700,
            barnorm="percent",
            legend_title_text="",
            legend=dict(font=dict(size=22))
        )

        # Displaying Chart
        with st.columns([0.5, 1.5, 0.5])[1]:
            st.plotly_chart(fig, use_container_width=True)

        # Displaying Insights & SQL Used
        st.caption(
            "Earlier positions (UTG, MP) fold more often but also raise aggressively when they do enter a hand, "
            "reflecting a cautious yet decisive approach due to acting first. As expected, later positions (CO, BTN) "
            "raise more frequently, leveraging position to apply pressure. The small blind (SB) has a moderate balance of actions, "
            "while the big blind (BB) has the highest check rate—given that it often sees the flop for free when no one raises. "
            "Overall, this chart highlights how position heavily influences preflop strategy."
        )
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
                ),
                action_type_counts AS(
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

    # Barchart - Average Winnings by Table Position
    with st.container(border=True):
        st.subheader("Average Winnings by Table Position")
        st.caption("The Average $ Won per Hand Played for every Position across stake levels")

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

        # Preapering data for chart
        df_avgwinnings_filtered = df_avgwinnings[(df_avgwinnings['seat_count'] == seat_count) & (df_avgwinnings['variant'] == variant)].copy()
        df_avgwinnings_filtered['position'] = df_avgwinnings_filtered['position'].replace(custom_labels)
        df_avgwinnings_filtered['position'] = pd.Categorical(df_avgwinnings_filtered['position'], categories=position_order, ordered=True)
        df_avgwinnings_filtered = df_avgwinnings_filtered.sort_values(by="position")

        # Creating & Styling the Chart
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
        # Displaying chart
        st.plotly_chart(fig)

        # Displaying insights & SQL used
        st.caption(
            "From left to right, the early positions (UTG, MP, CO) usually see moderate gains, while the Button (BTN) "
            "often stands out with the highest average winnings, thanks to acting last and facing fewer opponents. "
            "In contrast, both blinds (SB and BB) frequently show net losses, reflecting their forced bets and weaker positional advantage. "
            "Overall, these trends reinforce how critical position is in poker—later seats typically have the edge in decision-making and pot control, "
            "while the blinds bear the cost of mandatory bets and more complex postflop scenarios."
        )
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
                JOIN game_types_info AS g
                ON pg.game_id = g.game_id
                GROUP BY pg.position, g.seat_count, g.variant; """
            st.code(query, "sql")

    # Barchart - VPIP% & PFR% by Table Position
    with st.container(border=True):
        st.subheader("VPIP & PFR Percentages by Table Position")
        st.caption("Bar chart that shocases how often players voluntarily put money into the pot (VPIP) versus how frequently they raise preflop (PFR), broken down by each seat at the table")

        # Slider & Dropdown for stake level selection
        columns = st.columns([0.1, 1, 0.1, 1, 0.1])
        with columns[1]:
            variant = st.select_slider('Stake Level', [25, 50, 100, 200, 400, 600, 1000], value=1000, key="vpip_pfr_sld")
        with columns[3]:
            seat_count = st.selectbox("Table Type (Seat Count)", [6, 9], key="vpip_pfr_slctbox")
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

        # Preparing data for chart
        df_filtered = df_vpip_pfr_percentages[(df_vpip_pfr_percentages['seat_count'] == seat_count) & (df_vpip_pfr_percentages['variant'] == variant)].copy()
        df_filtered['position'] = df_filtered['position'].replace(custom_labels)
        df_filtered['position'] = pd.Categorical(df_filtered['position'], categories=position_order, ordered=True)
        df_filtered = df_filtered.sort_values(by="position")
        df_filtered.rename(columns={"vpip_percentage": "VPIP %", "pfr_percentage": "PFR %"}, inplace=True)

        # Creating & Styling chart
        fig = px.bar(
            df_filtered,
            x="position",
            y=["VPIP %", "PFR %"],
            barmode="group",
            labels={"position": "Position", "value": "Percentage (%)"},
            height=750,
            color_discrete_sequence=["#D1495B", "#577399"]
        )
        fig.update_layout(
            font=dict(size=14),
            xaxis=dict(tickfont_size=18, title_font_size=18),
            yaxis=dict(tickfont_size=18, title_font_size=18),
            legend_title_text="",
            legend=dict(
                font=dict(size=22),
                orientation="h",
                bgcolor="#171C26",
                bordercolor="#D3FFF3",
                borderwidth=0.2,
                x=0.85,
                y=0.9,
                xanchor="left",
                yanchor="top"
            )
        )
        fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside", textfont_size=20,)

        # Displaying chart
        st.plotly_chart(fig)

        # Displaying insights & SQL used
        st.caption(
            "Observing these numbers, you’ll notice that players in late positions (like the cutoff and button) often raise more before the flop. "
            "They’re in the best spots to steal blinds and face fewer opponents after them. Early positions, on the other hand, might have moderate or even higher VPIP "
            "(they see more flops), but they’re less likely to open-raise because they have to worry about the rest of the table acting behind them. "
            "Interestingly, the small blind sometimes shows a high PFR too, possibly from the pressure to defend or grab the pot immediately. "
            "Meanwhile, the big blind tends to be more selective, focusing on hands that can stand up to raises."
        )
        with st.expander("SQL Query Used"):
            query = """
                WITH sorted_actions AS (
                    SELECT DISTINCT ON (a.game_id, a.position)
                        a.position,
                        a.action_type,
                        gti.variant,
                        gti.seat_count
                    FROM actions AS a
                    JOIN game_types_info AS gti
                    ON a.game_id = gti.game_id
                    WHERE a.round = 0 AND a.amount > 0
                ),
                vpip_pfr_counts AS (
                    SELECT
                        a.variant,
                        a.seat_count,
                        a.position,
                        COUNT(*) AS vpip_actions_performed,
                        SUM(CASE WHEN action_type = 'cbr' THEN 1 ELSE 0 END) AS pfr_actions_performed
                    FROM sorted_actions AS a
                    GROUP BY a.position,a.variant,a.seat_count
                    ORDER BY a.variant, a.seat_count, a.position
                ),
                total_hands_dealt AS(
                    SELECT
                        gti.variant,
                        gti.seat_count,
                        pg.position,
                        COUNT(*) AS total_hands
                    FROM players_games AS pg
                    JOIN game_types_info AS gti
                    ON pg.game_id = gti.game_id
                    GROUP BY pg.position, gti.variant, gti.seat_count
                )
                SELECT
                    a.*,
                    thd.total_hands,
                    ROUND(a.vpip_actions_performed * 100.0 / thd.total_hands, 2) AS vpip_percentage,
                    ROUND(a.pfr_actions_performed * 100.0 / thd.total_hands, 2) AS pfr_percentage
                FROM vpip_pfr_counts AS a
                JOIN total_hands_dealt AS thd
                    ON thd.position = a.position
                    AND thd.variant = a.variant
                    AND a.seat_count = thd.seat_count"""
            st.code(query, "sql")

# TAB 3 - Trends Across Stake Levels
with tabs[2]:

    # Linechart - VPIP, PFR, 3BET Percentages across stake levels
    with st.container(border=True):
        st.subheader(f"VPIP, PFR & 3BET Percentages By Stake Level")
        st.caption("fill")  # TODO Fill

        # Slider & Dropdown for data selection
        columns = st.columns([0.1, 1, 0.1, 1, 0.1])
        with columns[1]:
            seat_count = st.selectbox("Table Type (Seat Count)", [9, 6], key="trends_across_stakes_slctbx")

        # Filtering & Preparing Data for chart
        df_filtered = df_vpip_pfr_3bet_across_stakes[(df_vpip_pfr_3bet_across_stakes['seat_count'] == seat_count)]
        df_filtered.rename(columns={"vpip": "VPIP %", "pfr": "PFR %", "threebet": "3BET %"}, inplace=True)

        # Creating & Styling chart
        fig = px.line(
            df_filtered,
            x="variant",
            y=["VPIP %", "PFR %", "3BET %"],
            labels={"variant": "Stake Level", "Value": "Percentages (%)"},
            color_discrete_sequence=["#74A57F",  "#577399", "#D1495B"],
            markers=True,
        )
        fig.update_layout(
            height=750,
            xaxis_type="category",
            legend_title_text="",
            legend=dict(
                font=dict(size=24),
                bgcolor="#171C26",
                bordercolor="#D3FFF3",
                borderwidth=0.2,
            ),
        )
        for trace in fig.data:
            trace.mode = "lines+markers+text"
            trace.text = [f'{y:.2f}' for y in trace.y]
            trace.textposition = 'top center'
            if trace.name == "VPIP %":
                trace.visible = "legendonly"
        fig.update_traces(line=dict(width=6), marker=dict(size=14), texttemplate='%{y:.2f}%', textposition='top center', textfont=dict(size=16))

        # Displaying Chart
        st.plotly_chart(fig)

        # Insights & SQL used
        st.caption("""
        On both 9 & 6 player tables, the data reveals a marked increase in both 3Bet and PFR percentages as stakes rise, 
        nearly doubling from the lowest to the highest levels.
        This suggests that higher-stakes play is characterized by a more aggressive and confident approach, 
        likely reflecting more refined skills and decisiveness.
        In contrast, VPIP remains fairly stable across all stakes, indicating that while players aren’t 
        necessarily entering more pots at higher stakes, they do engage more aggressively when they do.
        Overall, these trends underscore how table format and stake level influence preflop strategy, with 
        full-ring games displaying a more pronounced shift toward aggression at elevated stakes.
        """)
        with st.expander("SQL Query Used"):
            query = """
                WITH sorted_actions AS(
                    -- Get actions for round 0, calculate the running total of raises per game across all actions. add variant and seat_count for each action
                    SELECT 
                        gti.variant,
                        gti.seat_count,
                        a.game_id,
                        a.action_id,
                        a.position,
                        a.action_type,
                        a.amount,
                        SUM(CASE WHEN action_type = 'cbr' THEN 1 ELSE 0 END) OVER (PARTITION BY a.game_id ORDER BY a.action_id) AS raises_per_game
                    FROM actions AS a
                    JOIN game_types_info AS gti
                    ON a.game_id = gti.game_id
                    WHERE round = 0
                ),
                numbered_raises AS (
                    -- For each game number all the raises that happened pre-flop in chronological order 
                    SELECT
                        seat_count,
                        variant,
                        ROW_NUMBER () OVER (PARTITION BY a.game_id ORDER BY a.action_id) AS raise_number
                    FROM sorted_actions AS a
                    WHERE action_type = 'cbr'
                ),
                pfr_threebet_counts AS (
                    -- Summarize counts of first raises (PFR) and second raises (threebet) by variant and seat_count.
                    SELECT
                        seat_count,
                        variant,
                        SUM(CASE WHEN raise_number=1 THEN 1 ELSE 0 END) AS pfr_count,
                        SUM(CASE WHEN raise_number=2 THEN 1 ELSE 0 END) AS threebet_count
                    FROM numbered_raises
                    GROUP BY seat_count, variant
                ),
                opportunities_to_threebet_by_variant AS(
                    -- For each game, count the threebet opportunities provided and sum them as threebet_opportunity_count. Then aggregate by variant and seat_count.
                    SELECT
                        seat_count,
                        variant,
                        SUM (threebet_opportunity_count) AS opportunities_to_threebet
                    FROM (
                        SELECT 
                            game_id,
                            MAX(seat_count) AS seat_count,
                            MAX(variant) AS variant,
                            CASE 
                                WHEN SUM(CASE WHEN raises_per_game = 1 THEN 1 ELSE 0 END) > 0
                                THEN SUM(CASE WHEN raises_per_game = 1 THEN 1 ELSE 0 END) - 1
                                ELSE 0 END AS threebet_opportunity_count
                        FROM sorted_actions
                        GROUP BY game_id ) AS threebet_opportunites_per_game
                    GROUP BY seat_count, variant
                ),
                vpip_counts_table AS(
                    -- Counts total vpip actions and aggregates by variant & seat_count
                    SELECT 
                        variant,
                        seat_count,
                        COUNT(DISTINCT (game_id, sorted_actions.position)) AS vpip_count
                    FROM sorted_actions
                    WHERE amount > 0 
                    GROUP BY variant, seat_count
                ),
                total_count_table AS(
                    -- Counts the total number of hands per variant and seat_count.
                    SELECT 
                        gti.variant,
                        gti.seat_count,
                        COUNT(*) AS total_count
                    FROM players_games AS pg
                    JOIN game_types_info AS gti
                    ON pg.game_id = gti.game_id
                    GROUP BY gti.variant, gti.seat_count
                ),
                complete_counts AS(
                    -- Joins all the caclulated counts into one table.
                    SELECT 
                        og.variant,
                        og.seat_count,
                        vc.vpip_count,
                        og.pfr_count,
                        og.threebet_count,
                        ott.opportunities_to_threebet,
                        tc.total_count
                    FROM pfr_threebet_counts AS og
                    JOIN vpip_counts_table AS vc
                    ON vc.variant = og.variant AND vc.seat_count = og.seat_count
                    JOIN opportunities_to_threebet_by_variant AS ott
                    ON ott.variant = og.variant AND ott.seat_count = og.seat_count
                    JOIN total_count_table AS tc
                    ON tc.variant = og.variant AND tc.seat_count = og.seat_count
                )
                SELECT
                    -- compute percentages for VPIP, PFR, and threebet using the complete_counts table.
                    seat_count,
                    variant,
                    ROUND(vpip_count * 100.0 / total_count, 2) AS vpip,
                    ROUND(pfr_count * 100.0 / total_count, 2) AS pfr,
                    ROUND(threebet_count * 100.0 / opportunities_to_threebet, 2) AS threebet
                FROM complete_counts
                WHERE total_hands > 1000
                ORDER BY seat_count, variant
                """
            st.code(query, 'sql')
