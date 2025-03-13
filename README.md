# Poker Data Analysis

This project analyzes over 20 million poker hand histories stored in the Poker Hand History Standard (PHHS) format. Using PostgreSQL for data storage and SQL for data aggregation, the data is cleaned and transformed with Python and visualized interactively using Plotly and Streamlit.

## Demo
you can acess the live demo here: https://poker-data-analysis.streamlit.app/



## Technologies
- **Database:** PostgreSQL & SQL
- **Data Processing:** Python & Pandas
- **Visualization:** Plotly & Streamlit


## Getting Started

Follow these steps to set up and run the database locally:

1. **Download the Dataset:**  
   - Visit [Zenodo](https://zenodo.org/records/13997158) and download the collection of `.phhs` files. These files contain the raw poker hand histories required for the analysis.

2. **Set Up the Database Schema:**  
   - Use the provided `db_setup.sql` script to create the necessary PostgreSQL database schema.  
   - You can run the script using a PostgreSQL client or via the command line. For example:
     ```bash
     psql -U <your_username> -d <your_database> -f db_setup.sql
     ```

3. **Configure Environment Variables:**  
   - Rename the `.env.example` file to `.env`.  
   - Open the `.env` file and update the fields with your database credentials and other required settings.

4. **Populate the Database:**  
   - Ensure your virtual environment is activated and that all dependencies are installed (e.g., using `uv sync`)
   - Run the `population_script.py` to load the dataset into your database:
     ```bash
     python ./population_script.py
     ```