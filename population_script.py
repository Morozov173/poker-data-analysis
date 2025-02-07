import pandas as pd
import re
import ast
import mysql.connector
from mysql.connector import Error
from treys import Card, Evaluator
import itertools
from pathlib import Path
import time
from itertools import chain
import logging
from collections import Counter, deque
import gc


# Checks if a given string is a python literal
def is_literal(string):
    try:
        ast.literal_eval(string)
        return True
    except (ValueError, SyntaxError):
        return False


# Checks if a string is a number
def is_number(string):
    try:
        float(string)  # Attempt to convert the string to a float
        return True
    except ValueError:
        return False


# recieves a filepath to a phhs format file and returns a list where each item in the list is a single hand in dictionary format
def read_hands_from_phhs(file_path):

    # Reads file into memory via the given fila path parrameter

    with open(file_path, 'r') as file:
        content = file.read()

    # Splits all the hands in the file into seperate list items
    content = re.split(r'\[\d+\]', content)
    del content[0]
    content = [block.strip() for block in content]

    # Converts each hand into a dictionary format
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

        
        # Handles missing seat_count parameter to determine if it's 6Max or 9Max - Fixes hand
        if 'seat_count' not in temp_hand:
            if max(temp_hand['seats']) > 9:
                logger.warning(f"{str(current_file_path)[-50:]}:HAND {counter}:seat_count variable exceeded 9.")
                counter += 1
                continue
            if max(temp_hand['seats']) <= 6:
                temp_hand['seat_count'] = 6
            else:
                temp_hand['seat_count'] = 9

        

        # Handles an invalid amount of players in a game - Drops hand
        if len(temp_hand['players']) > 9:
            logger.warning(f"{str(current_file_path)[-50:]}:HAND {counter}:length of the players list exceeded 9.")
            counter += 1
            continue
        # Validates min_bet value by checking the min bet value of the two previus hands
        
        # Finds most common min bet value and handles min_bet values that don't match the common one
        if counter < 24:
            min_bet_occurrences[temp_hand['min_bet']] += 1
        if counter == 25:
            most_common_min_bet = min_bet_occurrences.most_common(1)[0][0]

        # Checks that the min_bet value is matching the most common - Fixes hand
        if most_common_min_bet != None and temp_hand['min_bet'] != most_common_min_bet:
            logger.warning(f"{str(current_file_path)[-50:]}:HAND {counter}: min_bet value is invalid it's {temp_hand['min_bet']} While the most common one is {most_common_min_bet}.")
            temp_hand['min_bet'] = most_common_min_bet

        # Handles invalid starting_stcaks list - Fixes hand
        if temp_hand['blinds_or_straddles'][1] != temp_hand['min_bet'] or temp_hand['blinds_or_straddles'][0] == 0 or temp_hand['blinds_or_straddles'][0] == temp_hand['blinds_or_straddles'][1]:
            logger.warning(f"{str(current_file_path)[-50:]}:HAND {counter}: List 'blinds_straddles' is invalid. big_blind value doesn't match min_bet value {temp_hand['blinds_or_straddles']}.")

            if temp_hand['min_bet'] == 0.25:
                temp_hand['blinds_or_straddles'][1] = temp_hand['min_bet']
                temp_hand['blinds_or_straddles'][0] = 0.10
            else:
                temp_hand['blinds_or_straddles'][1] = temp_hand['min_bet']
                temp_hand['blinds_or_straddles'][0] = int(temp_hand['min_bet'] / 2) if (temp_hand['min_bet'] / 2).is_integer() else round(temp_hand['min_bet'] / 2, 2)

        # Handles missing starting stacks values - Fixes hand
        if 'inf' in temp_hand['starting_stacks']:
            default_starting_stack = temp_hand['min_bet'] * 100
            temp_hand['starting_stacks'] = [default_starting_stack for starting_stack in temp_hand['players']]

        list_of_hands.append(temp_hand)
        counter += 1

    return list_of_hands


# Recieves the starting bets (small and big blinds) and the number of players
# returns a bet array where the columns are the players and each row is the bets they performed on the corresponding round
def create_bets_array(starting_blinds, amount_of_players):
    columns = [f'p{i+1}' for i in range(amount_of_players)]
    df = pd.DataFrame(
        [starting_blinds] + [[0.00] * amount_of_players] * 3, columns=columns)
    return df


# recieves hole cards of players and the board and returns the winners of the hand
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


# Recieves needed info about how a hand went and returns a list with the stacks the players finished the hand with
def calculate_finishing_stacks(df_player_bets, starting_stacks, hole_cards, community_cards, last_to_bet):
    total_bets = df_player_bets.sum()
    finishing_stacks = pd.Series(
        starting_stacks, index=total_bets.index) - total_bets

    # Hand didn't go to showdown, last to bet won.
    if hole_cards.count('????') == amount_of_players:
        finishing_stacks.iloc[int(last_to_bet[1])-1] += total_bets.sum()
        print('Hand didnt go to shodown')

    # Only one player has his cards revealed meaning he won for sure
    elif amount_of_players - hole_cards.count('????') == 1:
        for i, starting_hand in enumerate(hole_cards):
            if starting_hand != '????':
                finishing_stacks.iloc[i] += total_bets.sum()
        print('Only one player has his cards revealed meaning he won for sure')

    # Hand went to showdown
    else:
        if community_cards[0] is None:
                logger.error(f"{str(current_file_path)[-50:]}:HAND {hand_counter+1}:Went to showdown without community cards.")
                return starting_stacks
        winners = find_winning_hands(hole_cards, community_cards)
        paid_to_winner = total_bets.sum() / sum(winners)
        for i, winner in enumerate(winners):
            if winner is True:
                finishing_stacks.iloc[i] += paid_to_winner
        print(f'Hand went to showdown! and p{i+1} won!')

    return finishing_stacks


# queries the game_types table from the pokerhands database and converts it into a dataframe
def load_game_types_from_db():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
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
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
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


# returns the highest game_id in the games table of the database. if null returns 1
def fetch_game_id():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
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
        print(f"current game_id is: {game_id+1}")
        return game_id+1
    else:
        print(f"current game_id is: 1")
        return 1


# Exctracts file path from info.log starting processing line
def extract_file_path(string):
    starting_index =  string.find(':') + 1
    if starting_index == 0:
        logger.error(f"Couldn't find file_path in the provided string in the exctract_file_path function: {string}")
        return None
    ending_index = string.find(':', starting_index+2)
    file_path = string[starting_index:ending_index]
    return file_path.strip()


# set ups the regular logger. outputs warnings+ and info into two seperate files. and another file for tracking performance
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




# Preserve the last 15 lines of the previous `info.log` file before resetting it
with open("info.log", 'r') as file:
    last_lines = deque(file, 25) # Read last 25 log lines


# Save the preserved log lines into a separate file (`last_processed_file`) for later use
with open('last_processed_file', "w") as file:
    file.writelines(last_lines)

# Reinitialize the loggers (this will reset all '.log' files)
logger, performance_logger = set_up_loggers()


try:
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="5542",
        database="pokerhands_db"
    )
except Exception as e:
    logger.exception("Couldn't connect to the database.")
    print("Couldn't connect to the database.")
    print(e)

cursor = connection.cursor()
# Query to insert a row into the games table
insert_games_query = """
INSERT INTO games (game_type_id, amount_of_players, flop, turn, river, final_pot)
VALUES (%s, %s, %s, %s, %s, %s)
"""
# Query to insert a row into the players_games table
insert_players_games_query = """
INSERT INTO players_games (game_id, player_id, starting_stack, position, hand, winnings)
VALUES (%s, %s, %s, %s, %s, %s) """

# Query to insert a row into the actions table
insert_actions_query = """
INSERT INTO actions (game_id, action_id, position, action_type, round, amount, pot_size)
VALUES (%s, %s, %s, %s, %s, %s, %s) """

# 
start_time = time.time()
current_file_path = None
hand_counter = 0
root_dir = Path(r"D:\Programming\Poker Hands Dataset Zendoo\handhq")
game_id = fetch_game_id()
logger.info(f"Starting game_id: {game_id}\n")
df_game_types = load_game_types_from_db()


# If the database is empty, start parsing from the first file.
# Otherwise, check the log for the last processed file:
# - If the last file was successfully inserted, start at the next file.
# - If it failed, retry from the same file
if game_id != 1:
    logger.info(f"Continuing population of database from the last inserted file")
    skip_first_file = False
    start_parsing = False
    for line in reversed(last_lines):
        print(f"The line is: {line}")
        if skip_first_file == False and "Finished processing file succesfully" in line:
            skip_first_file = True
            logger.info(f"Last file was processed succesfully")
            continue
        if "Started processing file" in line:
            print("Found last processed file")
            try:
                starting_file_path = Path(extract_file_path(line))
                logger.info(f"last file to start processing was {starting_file_path}")
            except Exception as e:
                logger.exception(f"Couldn't convert exctracted starting_file filepath to Path object")
            break # Stop after finding the last processed file
else:
    start_parsing = True # If the database is empty, start parsing from the first file
    logger.info(f"Starting database population from scratch.")

                
            

print(start_parsing)
files_read_counter = 0
# Iterate through all .phhs files in the root directory
for file_path in root_dir.rglob("*.phhs"):
    # Skip files until reaching the last processed file; then start parsing
    if not start_parsing:
        if starting_file_path == file_path:
            start_parsing = True
            logger.info(f"File: {starting_file_path} was found!")
            if skip_first_file:
                continue
        else:
            continue # Keep skipping until the correct file is found

    # if starting_file == file_path:
    #     found = True
    #     logger.info(f"File: {starting_file} was found!")
    #     #continue    # Include 'continue' only if you want to skip processing the starting file
    # if found is False:
    #     continue

    # Lists to store data for batch insertion into corresponding database tables
    actions = []        # Stores player actions for the 'actions' table
    games = []          # Stores game metadata for the 'games' table
    players_games = []  # Stores player-game relationships for the 'players_games' table
    players_static = [] # Stores cumulative player stats for the 'players_static' table

    logger.info(f"{file_path}:GAME_ID {game_id}: Started processing file.")
    file_parsed_succesfully = True # Flag to track if file processing was successful
    batch_start_time = time.time()
    current_file_path = file_path
    hand_counter = 0
    
    
    
    try: 
        hands = read_hands_from_phhs(file_path)
    except Exception as e:
        logger.exception(f"{file_path}:Couldn't read file.")
        print("Couldn't connect to the database.")
        print(e)
        continue

    for hand in hands:
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

        print(f"\n\ncurrent hand being parsed is number: {hand_counter+1} with game_id: {game_id}")
        print(f"game type id is: {game_type_id}")
        print(f"players in the hand are {hand['players']}")

        # Table: actions ✔
        # Functionality: Parses the 'action' of a given hand and exctracts relevant information and stores it.
        # in addition formats it into rows and stores them in the 'actions' variable to later be inserted into the SQL Database
        end_pot = starting_pot
        round_counter = 0
        action_id = 1
        last_highest_bet = bb_amount
        last_to_bet = 'p2' #If the hand finished with no bets then big blind position won
        df_players_bets = create_bets_array(hand['blinds_or_straddles'], amount_of_players)


        print("The actions for the current hand are as follows: ")
        for action in hand['actions'][amount_of_players:]:
            temp_action = [game_id, action_id, None,None, round_counter, None, end_pot]
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
                temp_action[5] = float(last_highest_bet - df_players_bets.loc[round_counter, position])
                # saves the call amount in the bets dataframe
                df_players_bets.loc[round_counter, position] += temp_action[5]
                end_pot += temp_action[5]

            # if it's a RAISE or RE-RAISE
            elif action_type == 'cbr':
                bet_amount = float(action[2])
                if bet_amount <= 0: # Checks for invalid data
                    logger.error(f"{str(current_file_path)[-50:]}:HAND {hand_counter+1}:GAME_ID {game_id}:ACTION_ID: {action_id} - Negative bet amount was made.")
                    unrecoverable_data_detected = True # Raises flag to skip the current hand
                    break
                actual_bet_amount = float(bet_amount - df_players_bets.loc[round_counter, position])
                last_highest_bet = bet_amount
                df_players_bets.loc[round_counter, position] = bet_amount
                temp_action[5] = bet_amount
                end_pot += actual_bet_amount
                last_to_bet = position

            # if it's a SHOWDOWN
            else:
                # Updates the players hole cards since they were revealed at showdown
                hole_cards[int(position[1])-1] = action[2]

            action_id += 1
            temp_action[-1] = float(temp_action[-1])
            print(temp_action)
            current_hand_actions.append(tuple(temp_action))




        # Skip the hand if it contains unrecoverable data issues
        if unrecoverable_data_detected:
            hand_counter += 1
            continue

        # Log and skip the hand if no player actions were recorded (indicating missing or invalid data)
        if not current_hand_actions:
            logger.error(f"{str(current_file_path)[-50:]}:HAND {hand_counter+1}:GAME_ID {game_id}:ACTION_ID: {action_id}: Corrupted data, no actions ocurred in hand")
            hand_counter += 1
            continue

        # Log and skip the hand if the flop contains fewer than 3 cards (invalid game state)
        if community_cards[0] is not None and len(community_cards[0]) != 6:
            logger.error(f"{str(current_file_path)[-50:]}:HAND {hand_counter+1}:Flop of less then 3 cards. {community_cards}")
            hand_counter += 1
            continue
        
        # If the data checks passed we add the actions to the data and continue
        actions.append(current_hand_actions)


        finishing_stacks = calculate_finishing_stacks(df_players_bets, hand['starting_stacks'], hole_cards, community_cards, last_to_bet)
        
        # Print-outs for debugging
        print(f"the community cards are: \n{community_cards}")
        print(f"the hole cards are: \n{hole_cards}")
        print(f"the bets made during the hand are: \n{df_players_bets}")
        print(f"starting stacks are: \n{hand['starting_stacks']}")
        print(f"finishing stacks are: \n{finishing_stacks}")



        # Table: games ✔
        # Functionality: Creates and Formats a 'games' tuple to be inserted as a row into the "games" table.
        games.append((game_type_id, amount_of_players,community_cards[0], community_cards[1], community_cards[2], float(end_pot)))


        # Table: players_games ✔, players_static ✔
        # Functionality: Creates and Formats a tuple for each player that was in the hand and inserts them into the
        # players_games list to be inserted into the players_games table.
        for i, player in enumerate(hand['players']):
            player_winnings = float(finishing_stacks.iloc[i] - hand['starting_stacks'][i])
            players_static.append(player)
            players_static.append(player_winnings)

            players_games_row = (game_id, player, hand['starting_stacks'][i], f'p{i+1}', hole_cards[i], float(finishing_stacks.iloc[i] - hand['starting_stacks'][i]))
            current_hand_players_games.append(players_games_row)

        players_games.append(current_hand_players_games) 
        
        hand_counter += 1
        game_id += 1 # end of hand loop


    try:
        # Insertion into games table
        cursor.executemany(insert_games_query, games)
        connection.commit()
        logger.info("games inserted succesfully")

        # Insertion into players_games table
        flattened_players_games = list(chain.from_iterable(players_games))
        cursor.executemany(insert_players_games_query, flattened_players_games)
        logger.info("players_games inserted succesfully")
        
        # Insertion into actions table
        flattened_actions = list(chain.from_iterable(actions))
        cursor.executemany(insert_actions_query, flattened_actions)
        logger.info("actions inserted succesfully")

        # Insertion into players_static table
        format_string = ', '.join(["(%s, %s)"] * (len(players_static)//2))
        insert_players_static_query = f"""INSERT INTO players_static (player_id, total_winnings)
        VALUES {format_string}
        ON DUPLICATE KEY UPDATE
        total_winnings = total_winnings + VALUES(total_winnings),
        hands_played =  hands_played + 1"""
        cursor.execute(insert_players_static_query, tuple(players_static))
        logger.info("players_static inserted succesfully")

        # End of processesing the current file
        connection.commit()
        logger.info("all tables were commited succesfully")
        
    except KeyboardInterrupt:
        connection.rollback()
        cursor.close()
        connection.close()
        print("Keyboard Interrupt: Cleaning up before exit...")
        end_time = time.time()
        performance_logger.info(f"total time to process {files_read_counter} batches took {(end_time - start_time)/3600:.2f} hours")
    except Exception as e:
        file_parsed_succesfully = False
        connection.rollback()
        logger.exception(f"{str(current_file_path)[-50:]}: Couldn't perform commit or execute into the database.")

    files_read_counter += 1

    
    # Logging and performance info for the file that was finished.
    batch_end_time = time.time()
    performance_logger.info(f"the number {files_read_counter} batch took {batch_end_time - batch_start_time:.2f} seconds")
    if file_parsed_succesfully:
        logger.info(f"Finished processing file succesfully.\n")
    else:
        logger.info(f"File processing was unsuccesful.\n")

    # if files_read_counter == 5:
    #     break





connection.commit()
cursor.close()
connection.close()

end_time = time.time()
performance_logger.info(f"total time to process {files_read_counter} batches took {(end_time - start_time)/3600:.2f} hours")

