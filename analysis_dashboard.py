import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


st.header("General Analysis")


st.subheader("Biggest Pots Won")
st.caption("Biggest pot won for 123123 $")
df_biggest_pots = pd.read_csv("dashboard_data/biggest_pots.csv")


st.subheader("Most profitable players")
st.caption("Some very insightful commentary")

st.divider()
st.header("Positional Analysis")
st.subheader("Average Winnings per Hand played by Position")
st.subheader("VPIP & PFR Percentages by Position")
st.subheader("3Bet Percetnages by Position")
