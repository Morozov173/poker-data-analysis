# Poker Data Analysis

This project analyzes over 20 million poker hand histories stored in the Poker Hand History Standard (PHHS) format. Using PostgreSQL for data storage and SQL for data aggregation, the data is cleaned and transformed with Python and visualized interactively using Plotly and Streamlit.

You can view the analysis dashboard here: LINK


If you want to run and re-create the databse youreself:
1. download the dataset of .phhs files here: https://zenodo.org/records/13997158
2. set up the database schema using the db_setup.sql script
3. update the fields at .env file using the .env.example file provided
4. populate the database by running using population_script.py


## Technologies
- **Database:** PostgreSQL & SQL
- **Data Processing:** Python & Pandas
- **Visualization:** Plotly & Streamlit
