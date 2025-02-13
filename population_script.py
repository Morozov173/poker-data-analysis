import pandas as pd
import re
import ast
import mysql.connector
import psycopg2
from treys import Card, Evaluator
import itertools
from pathlib import Path
import time
from itertools import chain
import logging
from collections import Counter
import multiprocessing
import concurrent.futures
import os

# Function to check if a given string is a valid Python literal (e.g., list, dict, etc.)
def is_literal(string):
    try:
        ast.literal_eval(string)
        return True
    except (ValueError, SyntaxError):
        return False


# Function to check if a string is a valid number (integer or float)
def is_number(string):
    try:
        float(string)  # Attempt to convert the string to a float
        return True
    except ValueError:
        return False


# Function to read a PHHS file and parse it into a list of hands in dictionary format
def read_hands_from_phhs(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into individual hands
    content = re.split(r'\[\d+\]', content)
    del content[0]
    content = [block.strip() for block in content]

    list_of_hands = []
    counter = 1
    most_common_min_bet = None
    min_bet_occurrences = Counter()
    for hand in content:
        temp_hand = {}

        for line in hand.split('\n'):
            key, value = line.split(' = ')
            if is_literal(value):
                temp_hand[key] = ast.literal_eval(value)
            else:
                temp_hand[key] = value



        # Track most common min_bet from first 25 hands
        if counter < 24:
            min_bet_occurrences[temp_hand['min_bet']] += 1
        if counter == 25:
            most_common_min_bet = min_bet_occurrences.most_common(1)[0][0]

        # Handles missing seat_count 
        if 'seat_count' not in temp_hand:
            if max(temp_hand['seats']) > 9:
                counter += 1
                continue
            if max(temp_hand['seats']) <= 6:
                temp_hand['seat_count'] = 6
            else:
                temp_hand['seat_count'] = 9

        # Skip hand if invalid number of players
        if len(temp_hand['players']) > 9:
            counter += 1
            continue
        
        # Fix mismatched min_bet value
        if most_common_min_bet != None and temp_hand['min_bet'] != most_common_min_bet:
            temp_hand['min_bet'] = most_common_min_bet

        # Handle invalid starting_stacks
        if temp_hand['blinds_or_straddles'][1] != temp_hand['min_bet'] or temp_hand['blinds_or_straddles'][0] == 0 or temp_hand['blinds_or_straddles'][0] == temp_hand['blinds_or_straddles'][1]:

            if temp_hand['min_bet'] == 0.25:
                temp_hand['blinds_or_straddles'][1] = temp_hand['min_bet']
                temp_hand['blinds_or_straddles'][0] = 0.10
            else:
                temp_hand['blinds_or_straddles'][1] = temp_hand['min_bet']
                temp_hand['blinds_or_straddles'][0] = int(temp_hand['min_bet'] / 2) if (temp_hand['min_bet'] / 2).is_integer() else round(temp_hand['min_bet'] / 2, 2)

        # Handle missing starting stack values
        if 'inf' in temp_hand['starting_stacks']:
            default_starting_stack = temp_hand['min_bet'] * 100
            temp_hand['starting_stacks'] = [default_starting_stack for starting_stack in temp_hand['players']]


        list_of_hands.append(temp_hand)
        counter += 1

    return list_of_hands


# Function to create a bet array for a given hand, based on the number of players and blinds
def create_bets_array(starting_blinds, amount_of_players):
    columns = [f'p{i+1}' for i in range(amount_of_players)]
    df = pd.DataFrame(
        [starting_blinds] + [[0.00] * amount_of_players] * 3, columns=columns)
    return df


# Function to find the winning hand based on hole cards and community cards
def find_winning_hands(hole_cards, community_cards):
    encoded_community_cards = []
    encoded_community_cards.extend(re.findall("..?", community_cards[0]))
    encoded_community_cards.extend(community_cards[1:])
    encoded_community_cards = [
        Card.new(card) for card in encoded_community_cards if card is not None]

    winners = pd.Series([False]*len(hole_cards))
    scores = []
    evaluator = Evaluator()
    for hand in hole_cards:
        if hand == '????':
            scores.append(7462)  # Lowest possible score
        else:
            encoded_hand = [Card.new(hand[0:2]), Card.new(hand[2:])]
            scores.append(evaluator.evaluate(
                encoded_community_cards, encoded_hand))

    winners = pd.Series(scores) == pd.Series(scores).min()
    return winners


# Function to calculate finishing stacks after all actions are processed for a given hand
def calculate_finishing_stacks(df_player_bets, starting_stacks, hole_cards, community_cards, last_to_bet, amount_of_players):
    total_bets = df_player_bets.sum()
    finishing_stacks = pd.Series(
        starting_stacks, index=total_bets.index) - total_bets

    # Hand didn't go to showdown, last to bet won.
    if hole_cards.count('????') == amount_of_players:
        finishing_stacks.iloc[int(last_to_bet[1])-1] += total_bets.sum()
        #print('Hand didnt go to shodown')

    # Only one player has his cards revealed meaning he won for sure
    elif amount_of_players - hole_cards.count('????') == 1:
        for i, starting_hand in enumerate(hole_cards):
            if starting_hand != '????':
                finishing_stacks.iloc[i] += total_bets.sum()
        #print('Only one player has his cards revealed meaning he won for sure')

    # Hand went to showdown
    else:
        if community_cards[0] is None:
                print(f"Hand went to showdown without community cards.")
                return starting_stacks
        winners = find_winning_hands(hole_cards, community_cards)
        paid_to_winner = total_bets.sum() / sum(winners)
        for i, winner in enumerate(winners):
            if winner is True:
                finishing_stacks.iloc[i] += paid_to_winner
        #print(f'Hand went to showdown! and p{i+1} won!')

    return finishing_stacks


# queries the game_types table from the pokerhands database and converts it into a dataframe
def load_game_types_from_db():
    connection = psycopg2.connect(
        host="localhost",
        user="postgres",
        password="5542",
        database="pokerhands_db"
    )
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM game_types")

    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['game_type_id', 'seat_count', 'sb_amount', 'bb_amount', 'currency', 'variant'])
    df['sb_amount'] = df['sb_amount'].astype(float).round(2)
    df['bb_amount'] = df['bb_amount'].astype(float).round(2)
    cursor.close()
    connection.close()
    return df


# Recieves the existing game_types table from sql as a dataframe and selects the corresponding game_type_id for the given hand
# if it exists, if it doesn't adds it to the database
def find_game_type(df_game_types, seat_count, sb_amount, bb_amount, currency):

    game_type = [seat_count, sb_amount, bb_amount, currency, bb_amount*100]

    # Checks if the game_type of this hand exists
    matching_game_type_row = df_game_types[df_game_types.iloc[:, 1:].eq(
        game_type).all(axis=1)]

    if matching_game_type_row.empty:  # If the game type doesnt exist it is added to the dataframe and to the SQL Database
        new_game_type_id = df_game_types['game_type_id'].max()+1
        new_game_type = game_type
        new_game_type.insert(0, new_game_type_id)
        df_game_types.loc[len(df_game_types)] = game_type

        game_type[0] = game_type[0]
        connection = psycopg2.connect(
            host="localhost",
            user="postgres",
            password="5542",
            database="pokerhands_db"
        )
        cursor = connection.cursor()
        cursor.execute("""INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
            VALUES (%s, %s, %s, %s, %s)""", tuple([int(seat_count), float(sb_amount), float(bb_amount), str(currency), int(bb_amount*100)]))
        connection.commit()
        cursor.close()
        connection.close()
        return int(new_game_type_id)
    else:  # If it does we return the gametype id of this game type
        game_type_id = int(matching_game_type_row['game_type_id'].iloc[0])
        return int(game_type_id)


# Function to fetch the highest game_id from the games table, or return 1 if the table is empty
def fetch_game_id():
    connection = psycopg2.connect(
        host="localhost",
        user="postgres",
        password="5542",
        database="pokerhands_db"
    )
    cursor = connection.cursor()
    cursor.execute("SELECT MAX(game_id) FROM games")
    game_id = cursor.fetchall()
    game_id = game_id[0][0]

    cursor.close()
    connection.close()

    if game_id:
        #print(f"current game_id is: {game_id+1}")
        return game_id+1
    else:
        #print(f"current game_id is: 1")
        return 1


# Function to parse the file path from a string
def extract_file_path(string):
    starting_index =  string.find(':') + 1
    if starting_index == 0:
        print(f"Couldn't find file_path in the provided string in the exctract_file_path function: {string}")
        return None
    file_path = string[starting_index:]

    return file_path.strip()


# Function to determine the starting file for parsing based on the last processed batch or if the database is empty
def determine_starting_file_for_parsing(game_id):
    if game_id != 1:
        start_parsing = False # If will find starting file first

        print(f"Continuing population of database from the last inserted file")
        skip_first_file = False
        
        with open("last_processed_file", 'r') as file:
            lines = file.readlines()

        if lines[1] == 'SUCCESFULL':
            starting_file_path = extract_file_path(lines[0])
            skip_first_file = True
            print(f"Last batch was inserted succesfully so starting file is the last file of that batch: {starting_file_path} AND skip_first_file is True")
            return start_parsing, starting_file_path, skip_first_file
        
        else:
            skip_first_file = False
            starting_file_path = extract_file_path(lines[0])
            print(f"Last batch was inserted UNsuccesfully so starting file is the first file of that batch: {starting_file_path} AND skip_first_file is False")
            return start_parsing, starting_file_path, skip_first_file
    else:
        start_parsing = True # If the database is empty(game_id = 1), start parsing from the first file
        # logger.info(f"Starting database population from scratch.")
        return start_parsing, None, None
       

# Function to set up loggers for tracking performance, warnings, and general info
def set_up_loggers():
    # Logger set up
    performance_logger = logging.getLogger("performance_logger")
    performance_logger.propagate = False
    performance_logger.setLevel(logging.DEBUG)
    logger = logging.getLogger("general_logger")
    logger.setLevel(logging.DEBUG)

    # Handlers
    performance_handler = logging.FileHandler('performance.log', mode='w')
    performance_handler.setLevel(logging.DEBUG)
    warning_handler = logging.FileHandler("warnings.log", mode = 'w')
    warning_handler.setLevel(logging.WARNING)
    info_handler = logging.FileHandler("info.log", mode = 'w')
    info_handler.setLevel(logging.INFO)

    # Filtering
    class OnlyInfoFilter(logging.Filter):
        def filter(self, record):
            return record.levelno == logging.INFO  # Only allow `INFO` logs

    info_handler.addFilter(OnlyInfoFilter())

    # Formatting
    formatter = logging.Formatter("%(levelname)s:%(message)s")
    warning_handler.setFormatter(formatter)
    info_handler.setFormatter(formatter)

    # Attach Handlers
    performance_logger.addHandler(performance_handler)
    logger.addHandler(warning_handler)
    logger.addHandler(info_handler)
    
    return logger, performance_logger

# # TODO finish comment for function
def create_generator_of_phhs_batches(root_dir, start_parsing, starting_file_path, skip_first_file,  batch_size = os.cpu_count()):
    batch = []
    if starting_file_path:
        starting_file_path = Path(starting_file_path)

    for file_path in root_dir.rglob("*.phhs"):
        if not start_parsing:
            if starting_file_path == file_path:
                start_parsing = True
                print(f"File: {starting_file_path} was found!")
                if skip_first_file:
                    continue
            else:
                continue # Keep skipping until the correct file is found


        batch.append(file_path)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# # TODO finish comment for function
def process_actions_in_hand(actions_of_hand, amount_of_players, community_cards, hole_cards, df_bets_in_hand, sb_amount, bb_amount, game_id):
    round_counter = 0
    action_id = 1
    pot = sb_amount + bb_amount # Intilize the starting pot amount
    last_highest_bet = bb_amount
    last_to_bet = 'p2' #If the hand finished with no bets then big blind position won
    unrecoverable_data_detected = False
    processed_actions_of_hand = []

    for action in actions_of_hand[amount_of_players:]:
            temp_action = [game_id, action_id, None, None, round_counter, None, pot]
            action = action.split()

            # If it's the FLOP, TURN or RIVER and next ROUND starts
            if action[1] == 'db':
                community_cards[round_counter] = action[2]
                round_counter += 1
                last_highest_bet = 0  # bet for the round is reset
                continue

            # If it's actions performed by players
            position = action[0]
            action_type = action[1]
            temp_action[2] = position  # Adds position of player (p1,p2...)
            temp_action[3] = action_type  # Adds the action type

            # If it's a FOLD
            if action_type == 'f':
                pass

            # If it's a CHECK
            elif action_type == 'cc' and last_highest_bet == 0:
                temp_action[5] = 0  # Adds amount 0

            # If it's a CALL
            elif action_type == 'cc':
                # Adds the amount that was called
                temp_action[5] = float(last_highest_bet - df_bets_in_hand.loc[round_counter, position])
                # saves the call amount in the bets dataframe
                df_bets_in_hand.loc[round_counter, position] += temp_action[5]
                pot += temp_action[5]

            # if it's a RAISE or RE-RAISE
            elif action_type == 'cbr':
                bet_amount = float(action[2])

                if bet_amount <= 0: # Checks for invalid data
                    print(f"UNRECOREVABLE DATA DETECTED!. A MINUS BET AMOUNT")
                    unrecoverable_data_detected = True # Raises flag to skip the current hand
                    return processed_actions_of_hand, unrecoverable_data_detected, last_to_bet, pot
                
                actual_bet_amount = float(bet_amount - df_bets_in_hand.loc[round_counter, position])
                last_highest_bet = bet_amount
                df_bets_in_hand.loc[round_counter, position] = bet_amount
                temp_action[5] = bet_amount
                pot += actual_bet_amount
                last_to_bet = position

            # if it's a SHOWDOWN
            else:
                # Updates the players hole cards since they were revealed at showdown
                hole_cards[int(position[1])-1] = action[2]

            action_id += 1
            temp_action[-1] = float(temp_action[-1])
            #print(temp_action)
            processed_actions_of_hand.append(tuple(temp_action))

    return processed_actions_of_hand, unrecoverable_data_detected, last_to_bet, pot


# # TODO finish comment for function
def process_hands(path_to_phhs_file, shared_game_id, lock):
    hands = read_hands_from_phhs(path_to_phhs_file)
    #print(f"IM PROCESS_HANDS THAT WORKS ON {path_to_phhs_file}! MY GAME ID IS {shared_game_id['game_id']}")
    actions = []        # Stores player actions for the 'actions' table
    games = []          # Stores game metadata for the 'games' table
    players_games = []  # Stores player-game relationships for the 'players_games' table
    players_static = [] # Stores cumulative player stats for the 'players_static' table


    hand_counter = 0
    for hand in hands:
        with lock:
            game_id = shared_game_id['game_id']
            shared_game_id['game_id'] = shared_game_id['game_id'] +1

        #print(f"IM PROCESS_HANDS THAT WORKS ON {path_to_phhs_file}! MY GAME ID IS {game_id}")
        unrecoverable_data_detected = False
        current_hand_actions = []
        current_hand_players_games = []

        amount_of_players = len(hand['players'])
        hole_cards = ['????'] * amount_of_players
        community_cards = [None, None, None]
        sb_amount = hand['blinds_or_straddles'][0]
        bb_amount = hand['blinds_or_straddles'][1]
        starting_pot = sum(hand['blinds_or_straddles'])

        game_type_id = find_game_type(df_game_types, hand['seat_count'], sb_amount, bb_amount, hand['currency_symbol'])

        # print(f"\n\ncurrent hand being parsed is number: {hand_counter+1} with game_id: {game_id}")
        # print(f"game type id is: {game_type_id}")
        # print(f"players in the hand are {hand['players']}")

        # Table: actions ✔
        # Functionality: Parses the 'action' of a given hand and exctracts relevant information and stores it.
        # in addition formats it into rows and stores them in the 'actions' variable to later be inserted into the SQL Database

        df_bets_in_hand = create_bets_array(hand['blinds_or_straddles'], amount_of_players)
        #print("The actions for the current hand are as follows: ")
        current_hand_actions, unrecoverable_data_detected, last_to_bet, end_pot = process_actions_in_hand(hand['actions'], amount_of_players, community_cards, hole_cards, df_bets_in_hand, sb_amount, bb_amount, game_id)

        

        # Log and skip the hand if no player actions were recorded (indicating missing or invalid data)
        if not current_hand_actions:
            #print(f"{str(current_file_path)[-50:]}:HAND {hand_counter+1}:GAME_ID {game_id}: Corrupted data, no actions ocurred in hand")
            unrecoverable_data_detected = True


        # Log and skip the hand if the flop contains fewer than 3 cards (invalid game state)
        if community_cards[0] is not None and len(community_cards[0]) != 6:
            #print(f"{str(current_file_path)[-50:]}:HAND {hand_counter+1}:Flop of less then 3 cards. {community_cards}")
            unrecoverable_data_detected = True
        
        # Skip the hand if it contains unrecoverable data issues
        if unrecoverable_data_detected:
            hand_counter += 1
            continue

        # If the data checks passed we add the actions to the data and continue
        actions.append(current_hand_actions)


        finishing_stacks = calculate_finishing_stacks(df_bets_in_hand, hand['starting_stacks'], hole_cards, community_cards, last_to_bet, amount_of_players)
        
        # Print-outs for debugging
        # print(f"the community cards are: \n{community_cards}")
        # print(f"the hole cards are: \n{hole_cards}")
        # print(f"the bets made during the hand are: \n{df_bets_in_hand}")
        # print(f"starting stacks are: \n{hand['starting_stacks']}")
        # print(f"finishing stacks are: \n{finishing_stacks}")



        # Table: games ✔
        # Functionality: Creates and Formats a 'games' tuple to be inserted as a row into the "games" table.
        # 
        games.append((game_id, game_type_id, amount_of_players,community_cards[0], community_cards[1], community_cards[2], float(end_pot)))


        # Table: players_games ✔, players_static ✔
        # Functionality: Creates and Formats a tuple for each player that was in the hand and inserts them into the
        # players_games list to be inserted into the players_games table.
        for i, player_id in enumerate(hand['players']):
            player_winnings = float(finishing_stacks.iloc[i] - hand['starting_stacks'][i])
            players_static.append((player_id, player_winnings))

            players_games_row = (game_id, player_id, hand['starting_stacks'][i], f'p{i+1}', hole_cards[i], float(finishing_stacks.iloc[i] - hand['starting_stacks'][i]))
            current_hand_players_games.append(players_games_row)
        players_games.append(current_hand_players_games) 
        
        hand_counter += 1

    print(f"FINISHED PROCESSING {path_to_phhs_file}") 
    return games, players_games, actions, players_static


# Inserts data into database using executemany #TODO add better commenting to function
def execute_bulk_insert(query, data, cursor = None, connection = None, commit = False, flatten = False):
    if cursor is None:
        try:
            connection =  psycopg2.connect(
            host='localhost',
            user="postgres",
            password="5542",
            database="pokerhands_db"
            )
            cursor = connection.cursor()
        except Exception as e:
            print("Couldn't connect to the database.")

    try:
        table_name = query.split()[2]
    except IndexError:
        table_name = "unknown_table"  # Fallback if query format is unexpected
        print(f"During insertion the passed query parameter was invalid")

    if flatten:
        data = list(chain.from_iterable(data))

    try: 
        cursor.executemany(query, data)
        if commit:
            connection.commit()
        print(f"Insertion into the {table_name} table completed succesfully")
    except Exception as e:
        connection.rollback()
        print(f"Insertion into the {table_name} table failed. reason: {e}")







df_game_types = load_game_types_from_db()

def main():
    start_time = time.time()

    # Set up connection to DB
    try:
        connection =  psycopg2.connect(
        host='localhost',
        user="postgres",
        password="5542",
        database="pokerhands_db"
    )
    except Exception as e:
        logger.exception("Couldn't connect to the database.")
        print("Couldn't connect to the database.")
        print(e)
    cursor = connection.cursor()
    cursor.execute("SET search_path TO testing, public;")
    logger, performance_logger = set_up_loggers()
    manager = multiprocessing.Manager()
    shared_game_id = manager.dict()
    shared_game_id['game_id'] = fetch_game_id()
    lock = manager.Lock()

    logger.info(f"MESSAGE: Most recent GAME_ID:{shared_game_id['game_id']} Fetched from database.")
    logger.info(f"MESSAGE: Amount of cpu cores detected - {os.cpu_count()}\n")

    root_dir = Path(r"D:\Programming\Poker Hands Dataset Zendoo\handhq")
    start_parsing, starting_file_path, skip_first_file = determine_starting_file_for_parsing(shared_game_id['game_id'])
    phhs_generator = create_generator_of_phhs_batches(root_dir, start_parsing, starting_file_path, skip_first_file)

    


    files_read_counter = 0
    batches_processed_counter = 1
    for batch in phhs_generator:
        logger.info(f"BATCH_ID {batches_processed_counter}:GAME_ID {shared_game_id['game_id']}:MESSAGE: Started Processing of batch")
        batch_start_time = time.time()
        file_parsed_succesfully = True

        # Lists to store data for batch insertion into corresponding database tables
        all_games = []          # Stores game metadata for the 'games' table
        all_players_games = []  # Stores player-game relationships for the 'players_games' table
        all_actions = []        # Stores player actions for the 'actions' table 
        all_players_static = [] # Stores cumulative player stats for the 'players_static' table

        # Multiprocesses the parsing and formatting of data
        with concurrent.futures.ProcessPoolExecutor() as executor:
            print(f"starting game_id shared variable is: {shared_game_id['game_id']}")
            futures = [executor.submit(process_hands, file_path, shared_game_id, lock) for file_path in batch]
            for future in futures:
                games, players_games, actions, players_static = future.result()

                all_games = list(itertools.chain(all_games, games))
                all_players_games = list(itertools.chain(all_players_games, players_games))
                all_actions = list(itertools.chain(all_actions, actions))
                all_players_static = list(itertools.chain(all_players_static, players_static))



        try:
            insert_games_query = """
            INSERT INTO games (game_id, game_type_id, amount_of_players, flop, turn, river, final_pot)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            execute_bulk_insert(insert_games_query, all_games, cursor, connection, commit=False, flatten=False)

            insert_players_games_query = """
            INSERT INTO players_games (game_id, player_id, starting_stack, position, hand, winnings)
            VALUES (%s, %s, %s, %s, %s, %s) """
            execute_bulk_insert(insert_players_games_query, all_players_games, cursor, connection, commit=False, flatten=True)

            insert_actions_query = """
            INSERT INTO actions (game_id, action_id, position, action_type, round, amount, pot_size)
            VALUES (%s, %s, %s, %s, %s, %s, %s) """
            execute_bulk_insert(insert_actions_query, all_actions, cursor, connection, commit=False, flatten=True)

            insert_players_static_query = """
            INSERT INTO players_static (player_id, total_winnings)
            VALUES (%s, %s)
            ON CONFLICT (player_id) 
            DO UPDATE 
            SET total_winnings = players_static.total_winnings + EXCLUDED.total_winnings,
                hands_played  = players_static.hands_played + 1
            """
            execute_bulk_insert(insert_players_static_query, all_players_static, cursor, connection, commit=False, flatten=False)

            connection.commit()
            print("All tables were commited succesfully")

        except KeyboardInterrupt:
            connection.rollback()
            cursor.close()
            connection.close()
            logger.exception(f"BATCH_ID {batches_processed_counter}:GAME_ID {shared_game_id['game_id']}:MESSAGE: Keyboard Interrupt detected. Cleaning up before exit.")
            end_time = time.time()

        except Exception as e:
            file_parsed_succesfully = False
            connection.rollback()
            logger.exception(f"BATCH_ID {batches_processed_counter}:GAME_ID {shared_game_id['game_id']}:MESSAGE: Insert or Commit raised an Error when executing into DB.")

        

        if file_parsed_succesfully:
            for file_path in batch:
                logger.info(f"Succesfully processed and inserted: {file_path}")
            logger.info(f"BATCH_ID {batches_processed_counter}:GAME_ID:{shared_game_id['game_id']}:MESSAGE: COMPLETED processing batch of files succesfully\n")

            with open('last_processed_file', "w") as file:
                file.writelines([f"LAST PROCESSED FILE OF BATCH:{batch[-1]}\n", "SUCCESFULL"]) 
        else:
            logger.info(f"BATCH_ID {batches_processed_counter}:GAME_ID:{shared_game_id['game_id']}:MESSAGE: FAILED processing batch of files.")
            with open('last_processed_file', "w") as file:
                file.writelines([f"FIRST PROCESSED FILE OF BATCH:{batch[0]}\n", "UNSUCCESFULL"])

        files_read_counter += len(batch)
        batches_processed_counter += 1
        performance_logger.info(f"BATCH_ID {batches_processed_counter}: Took {time.time() - batch_start_time:.2f} to process. Average time per file in batch was:{(time.time() - batch_start_time) / len(batch):.2f}")


        
    manager.shutdown        
    connection.commit()
    cursor.close()
    connection.close()

    end_time = time.time()
    performance_logger.info(f"total time to process {files_read_counter} batches took {(end_time - start_time)/3600:.2f} hours or {(end_time - start_time):.2f} seconds")
    logger.info(f"Succesfully inserted all hands into DB!")

if __name__ == "__main__":
    main()