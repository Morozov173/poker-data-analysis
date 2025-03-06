from typing import Any, Generator
import os
import pandas as pd
import re
import ast
import psycopg2
from treys import Card, Evaluator
import itertools
from pathlib import Path
import time
from itertools import chain
import logging
from collections import Counter
import multiprocessing
from multiprocessing import managers
import concurrent.futures
from dotenv import load_dotenv


# Returns True if the string is a valid Python literal; otherwise, False.
def is_literal(s: str) -> bool:
    try:
        ast.literal_eval(s)
        return True
    except (ValueError, SyntaxError):
        return False


# Returns True if the string converts to a float; otherwise, False.
def is_number(s: str) -> bool:
    try:
        float(s)  # Attempt to convert the string to a float
        return True
    except ValueError:
        return False


# Reads a PHHS file and parses its content into a list of hand dictionaries.
def read_hands_from_phhs(file_path: str) -> list[dict]:
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

        # Handles an invalid amount of actions
        if len(temp_hand['actions']) <= 2:
            counter += 1
            with open('DEBUGGING1.txt', 'a') as file:
                file.write(
                    f"FEWER THAN 2 ACTIONS IN {file_path} HAND NUMBER {counter}\n")
            continue

        # Handles missing/invalid seat_count
        if 'seat_count' not in temp_hand:
            if max(temp_hand['seats']) > 9:
                counter += 1
                continue
            if max(temp_hand['seats']) <= 6:
                temp_hand['seat_count'] = 6
            else:
                temp_hand['seat_count'] = 9
        elif temp_hand['seat_count'] < 6:
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
        correct_bb = temp_hand['min_bet']
        if correct_bb == 0.25:
            correct_sb = 0.10
        else:
            correct_sb = int(temp_hand['min_bet'] / 2) if (
                temp_hand['min_bet'] / 2).is_integer() else round(temp_hand['min_bet'] / 2, 2)
        if temp_hand['blinds_or_straddles'][1] != correct_bb or temp_hand['blinds_or_straddles'][0] != correct_sb:
            temp_hand['blinds_or_straddles'][1] = correct_bb
            temp_hand['blinds_or_straddles'][0] = correct_sb

        # Handle missing starting stack values
        if 'inf' in temp_hand['starting_stacks']:
            default_starting_stack = temp_hand['min_bet'] * 100
            temp_hand['starting_stacks'] = [
                default_starting_stack for starting_stack in temp_hand['players']]

        # Handles where p1 isn't the small blind. happnens in heads-up games only
        if len(temp_hand['players']) == 2 and temp_hand['actions'][2][0:2] == 'p2':
            temp_hand['players'].reverse()
            temp_hand['actions'] = [s.replace('p1', "TEMP_PLACEHOLDER").replace(
                'p2', "p1").replace("TEMP_PLACEHOLDER", 'p2') for s in temp_hand['actions']]

        list_of_hands.append(temp_hand)
        counter += 1

    return list_of_hands


# Creates a DataFrame of bets using starting blinds and the number of players.
def create_bets_array(starting_blinds: list[float], amount_of_players: int) -> pd.DataFrame:
    columns = [f'p{i+1}' for i in range(amount_of_players)]
    df = pd.DataFrame(
        [starting_blinds] + [[0.00] * amount_of_players] * 3, columns=columns)
    return df


# Determines winning hand(s) by evaluating hole cards against community cards.
def find_winning_hands(hole_cards: list[str], community_cards: list[str | None]) -> pd.Series:
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


# Calculates final stacks for players after all betting actions.
def calculate_finishing_stacks(df_player_bets: pd.DataFrame, starting_stacks: list[float], hole_cards: list[str], community_cards: list[str | None], last_to_bet: str, amount_of_players: int) -> pd.Series:
    total_bets = df_player_bets.sum()
    finishing_stacks = pd.Series(
        starting_stacks, index=total_bets.index) - total_bets

    # Hand didn't go to showdown, last to bet won.
    if hole_cards.count('????') == amount_of_players:
        finishing_stacks.iloc[int(last_to_bet[1])-1] += total_bets.sum()

    # Only one player has his cards revealed meaning he won for sure
    elif amount_of_players - hole_cards.count('????') == 1:
        for i, starting_hand in enumerate(hole_cards):
            if starting_hand != '????':
                finishing_stacks.iloc[i] += total_bets.sum()

    # Hand went to showdown
    else:
        if community_cards[0] is None:
            return starting_stacks

        winners = find_winning_hands(hole_cards, community_cards)
        paid_to_winner = total_bets.sum() / sum(winners)
        for i, winner in enumerate(winners):
            if winner is True:
                finishing_stacks.iloc[i] += paid_to_winner

    return finishing_stacks


# Loads the game_types table from the database into a DataFrame.
def load_game_types_from_db() -> pd.DataFrame:
    connection = psycopg2.connect(
        host=os.getenv("HOST"),
        user=os.getenv("USER"),
        password=os.getenv("PASSWORD"),
        database=os.getenv("DB_NAME")
    )
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM game_types")

    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=[
                      'game_type_id', 'seat_count', 'sb_amount', 'bb_amount', 'currency', 'variant'])
    df['sb_amount'] = df['sb_amount'].astype(float).round(2)
    df['bb_amount'] = df['bb_amount'].astype(float).round(2)
    cursor.close()
    connection.close()
    return df


# Returns the game_type_id matching the parameters; inserts a new type if needed.
def find_game_type(df_game_types: pd.DataFrame, seat_count: int, sb_amount: float, bb_amount: float, currency: str) -> int:

    game_type = [seat_count, sb_amount, bb_amount, currency, bb_amount*100]

    # Checks if the game_type of this hand exists
    matching_game_type_row = df_game_types[df_game_types.iloc[:, 1:].eq(
        game_type).all(axis=1)]

    if matching_game_type_row.empty:  # If the game type doesnt exist it is added to the dataframe and to the SQL Database
        with open('DEBUGGING2.txt', 'a') as file:
            file.write(
                f"A NEW GAME TYPE ID IS BEING CREATED FOR GAME TYPE: {game_type}\n")
        new_game_type_id = df_game_types['game_type_id'].max()+1
        new_game_type = game_type
        new_game_type.insert(0, new_game_type_id)
        df_game_types.loc[len(df_game_types)] = game_type

        game_type[0] = game_type[0]
        connection = psycopg2.connect(
            host=os.getenv("HOST"),
            user=os.getenv("USER"),
            password=os.getenv("PASSWORD"),
            database=os.getenv("DB_NAME")
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


# Retrieves the maximum game_id from the DB and returns the next available id.
def fetch_game_id() -> int:
    connection = psycopg2.connect(
        host=os.getenv("HOST"),
        user=os.getenv("USER"),
        password=os.getenv("PASSWORD"),
        database=os.getenv("DB_NAME")
    )
    cursor = connection.cursor()
    cursor.execute("SELECT MAX(game_id) FROM games")
    game_id = cursor.fetchall()
    game_id = game_id[0][0]

    cursor.close()
    connection.close()

    if game_id:
        return game_id+1
    else:
        return 1


# Extracts and returns a file path from a given string.
def extract_file_path(s: str) -> str:
    starting_index = s.find(':') + 1
    if starting_index == 0:
        print(
            f"Couldn't find file_path in the provided string in the exctract_file_path function: {s}")
        return None
    file_path = s[starting_index:]

    return file_path.strip()


# Decides where to begin parsing based on the last processed file
def determine_starting_file_for_parsing(game_id: int) -> tuple[bool, str | None, bool | None]:
    if game_id != 1:
        start_parsing = False  # If will find starting file first

        print(f"Continuing population of database from the last inserted file")
        skip_first_file = False

        with open("last_processed_file", 'r') as file:
            lines = file.readlines()

        if lines[1] == 'SUCCESFULL':
            starting_file_path = extract_file_path(lines[0])
            skip_first_file = True
            print(
                f"Last batch was inserted succesfully so starting file is the last file of that batch: {starting_file_path} AND skip_first_file is True")
            return start_parsing, starting_file_path, skip_first_file

        else:
            skip_first_file = False
            starting_file_path = extract_file_path(lines[0])
            print(
                f"Last batch was inserted UNsuccesfully so starting file is the first file of that batch: {starting_file_path} AND skip_first_file is False")
            return start_parsing, starting_file_path, skip_first_file
    else:
        # If the database is empty(game_id = 1), start parsing from the first file
        start_parsing = True
        return start_parsing, None, None


# Configures and returns two loggers for general and performance logging.
def set_up_loggers() -> tuple[logging.Logger, logging.Logger]:
    # Logger set up
    performance_logger = logging.getLogger("performance_logger")
    performance_logger.propagate = False
    performance_logger.setLevel(logging.DEBUG)
    logger = logging.getLogger("general_logger")
    logger.setLevel(logging.DEBUG)

    # Handlers
    performance_handler = logging.FileHandler('performance.log', mode='w')
    performance_handler.setLevel(logging.DEBUG)
    warning_handler = logging.FileHandler("warnings.log", mode='w')
    warning_handler.setLevel(logging.WARNING)
    info_handler = logging.FileHandler("info.log", mode='w')
    info_handler.setLevel(logging.INFO)

    # Filtering
    class OnlyInfoFilter(logging.Filter):
        def filter(self, record):
            return record.levelno == logging.INFO

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


# Yields batches of .phhs file paths from a directory starting at a specified file.
def create_generator_of_phhs_batches(root_dir: Path, start_parsing: bool, starting_file_path: str, skip_first_file: bool,  batch_size=os.cpu_count()) -> Generator[list[str], None, None]:
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
                continue  # Keep skipping until the correct file is found

        batch.append(file_path)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# Processes a hand’s actions, updating bets, pot, and tracking the last bet and errors.
def process_actions_in_hand(actions_of_hand: list[str], amount_of_players: int, community_cards: list[str | None], hole_cards: list[str], df_bets_in_hand: pd.DataFrame, sb_amount: float, bb_amount: float, game_id: int) -> tuple[tuple, bool, str, float]:
    round_counter = 0
    action_id = 1
    pot = sb_amount + bb_amount  # Intilize the starting pot amount
    last_highest_bet = bb_amount
    last_to_bet = 'p2'  # If the hand finished with no bets then big blind position won
    unrecoverable_data_detected = False
    processed_actions_of_hand = []

    for action in actions_of_hand[amount_of_players:]:
        temp_action = [game_id, action_id, None,
                       None, round_counter, None, pot]
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
            temp_action[5] = float(
                last_highest_bet - df_bets_in_hand.loc[round_counter, position])
            # saves the call amount in the bets dataframe
            df_bets_in_hand.loc[round_counter, position] += temp_action[5]
            pot += temp_action[5]

        # if it's a RAISE or RE-RAISE
        elif action_type == 'cbr':
            bet_amount = float(action[2])

            if bet_amount <= 0:  # Checks for invalid data
                unrecoverable_data_detected = True  # Raises flag to skip the current hand
                return processed_actions_of_hand, unrecoverable_data_detected, last_to_bet, pot

            actual_bet_amount = float(
                bet_amount - df_bets_in_hand.loc[round_counter, position])
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
        processed_actions_of_hand.append(tuple(temp_action))

    return processed_actions_of_hand, unrecoverable_data_detected, last_to_bet, pot


# Parses a PHHS file and returns data as four lists. Uses a Manager dict proxy and lock for safe multi-process sharing.
def process_hands(path_to_phhs_file: str, shared_game_id: managers.DictProxy, lock: managers.AcquirerProxy) -> tuple[list[tuple], list[list], list[list], list[tuple]]:
    hands = read_hands_from_phhs(path_to_phhs_file)

    actions = []        # Stores player actions for the 'actions' table
    games = []          # Stores game metadata for the 'games' table
    players_games = []  # Stores player-game relationships for the 'players_games' table
    players_static = []  # Stores cumulative player stats for the 'players_static' table

    hand_counter = 0
    for hand in hands:
        with lock:
            game_id = shared_game_id['game_id']
            shared_game_id['game_id'] = shared_game_id['game_id'] + 1

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

        df_bets_in_hand = create_bets_array(hand['blinds_or_straddles'], amount_of_players)

        current_hand_actions, unrecoverable_data_detected, last_to_bet, end_pot = process_actions_in_hand(
            hand['actions'], amount_of_players, community_cards, hole_cards, df_bets_in_hand, sb_amount, bb_amount, game_id)

        # Raise flag for the hand if no player actions were recorded (indicating missing or invalid data)
        if not current_hand_actions:
            unrecoverable_data_detected = True

        # Raise flag for the hand hand if the flop contains fewer than 3 cards (invalid game state)
        if community_cards[0] is not None and len(community_cards[0]) != 6:
            unrecoverable_data_detected = True

        # Skip the hand if it contains unrecoverable data issues
        if unrecoverable_data_detected:
            hand_counter += 1
            continue

        # If the data checks passed we add the actions to the data and continue
        actions.append(current_hand_actions)

        finishing_stacks = calculate_finishing_stacks(
            df_bets_in_hand, hand['starting_stacks'], hole_cards, community_cards, last_to_bet, amount_of_players)

        games.append((game_id, game_type_id, amount_of_players,
                     community_cards[0], community_cards[1], community_cards[2], float(end_pot)))

        # Creates the players_games and players_static lists to be inserted into DB.
        for i, player_id in enumerate(hand['players']):
            player_winnings = float(
                finishing_stacks.iloc[i] - hand['starting_stacks'][i])
            players_static.append((player_id, player_winnings))

            players_games_row = (game_id, player_id, hand['starting_stacks'][i], f'p{i+1}', hole_cards[i], float(
                finishing_stacks.iloc[i] - hand['starting_stacks'][i]))
            current_hand_players_games.append(players_games_row)
        players_games.append(current_hand_players_games)

        hand_counter += 1

    return games, players_games, actions, players_static


# Performs a bulk database insert using executemany with optional data flattening.
def execute_bulk_insert(query, data, connection=None, commit=False, flatten=False, close_connection=False):

    if connection is None:
        try:
            connection = psycopg2.connect(
                host=os.getenv("HOST"),
                user=os.getenv("USER"),
                password=os.getenv("PASSWORD"),
                database=os.getenv("DB_NAME")
            )

        except Exception as e:
            print("Couldn't connect to the database.")

    cursor = connection.cursor()
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
    finally:
        if close_connection:
            cursor.close()
            connection.close()


load_dotenv()
df_game_types = load_game_types_from_db()


def main():
    start_time = time.time()

    logger, performance_logger = set_up_loggers()
    manager = multiprocessing.Manager()
    shared_game_id = manager.dict()
    shared_game_id['game_id'] = fetch_game_id()
    lock = manager.Lock()

    logger.info(f"MESSAGE: Most recent GAME_ID:{shared_game_id['game_id']} Fetched from database.")
    logger.info(f"MESSAGE: Amount of cpu cores detected - {os.cpu_count()}\n")

    root_dir = Path(os.getenv("ROOT_DIR"))
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
        all_players_static = []  # Stores cumulative player stats for the 'players_static' table

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

        # Set up connection to DB.
        try:
            connection = psycopg2.connect(
                host=os.getenv("HOST"),
                user=os.getenv("USER"),
                password=os.getenv("PASSWORD"),
                database=os.getenv("DB_NAME")
            )
        except Exception as e:
            logger.exception("Couldn't connect to the database.")
            print("Couldn't connect to the database.")
            print(e)
        cursor = connection.cursor()

        # Insertion of all data into the appropriate tables of the DB.
        try:
            insert_games_query = """
            INSERT INTO games (game_id, game_type_id, amount_of_players, flop, turn, river, final_pot)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            execute_bulk_insert(insert_games_query, all_games, connection, commit=False, flatten=False, close_connection=False)

            insert_players_games_query = """
            INSERT INTO players_games (game_id, player_id, starting_stack, position, hand, winnings)
            VALUES (%s, %s, %s, %s, %s, %s) """
            execute_bulk_insert(insert_players_games_query, all_players_games, connection, commit=False, flatten=True, close_connection=False)

            insert_actions_query = """
            INSERT INTO actions (game_id, action_id, position, action_type, round, amount, pot_size)
            VALUES (%s, %s, %s, %s, %s, %s, %s) """
            execute_bulk_insert(insert_actions_query, all_actions, connection, commit=False, flatten=True, close_connection=False)

            insert_players_static_query = """
            INSERT INTO players_static (player_id, total_winnings)
            VALUES (%s, %s)
            ON CONFLICT (player_id) 
            DO UPDATE 
            SET total_winnings = players_static.total_winnings + EXCLUDED.total_winnings,
            hands_played  = players_static.hands_played + 1
            """
            execute_bulk_insert(insert_players_static_query, all_players_static, connection, commit=False, flatten=False, close_connection=False)

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
        performance_logger.info(
            f"BATCH_ID {batches_processed_counter}: Took {time.time() - batch_start_time:.2f} to process. Average time per file in batch was:{(time.time() - batch_start_time) / len(batch):.2f}")

    manager.shutdown
    connection.commit()
    cursor.close()
    connection.close()

    end_time = time.time()
    performance_logger.info(f"total time to process {files_read_counter} batches took {(end_time - start_time)/3600:.2f} hours or {(end_time - start_time):.2f} seconds")
    logger.info(f"Succesfully inserted all hands into DB!")


if __name__ == "__main__":
    main()
