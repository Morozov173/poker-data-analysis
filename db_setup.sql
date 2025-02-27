-- 1) Drop the database if it exists
DROP DATABASE IF EXISTS pokerhands_db;

-- 2) Create the database fresh
CREATE DATABASE pokerhands_db;

-- 3) Connect to the new database (psql meta-command)
\c pokerhands_db

------------------------------------------------------------------------------
-- All the following statements run within pokerhands_db
------------------------------------------------------------------------------

-- CREATE TABLE: game_types
CREATE TABLE game_types (
    game_type_id SERIAL PRIMARY KEY,
    seat_count SMALLINT NOT NULL,
    sb_amount DECIMAL(5,2) NOT NULL,
    bb_amount DECIMAL(5,2) NOT NULL,
    currency VARCHAR(4) NOT NULL DEFAULT '$',
    variant INT NOT NULL
);

-- CREATE TABLE: games
CREATE TABLE games (
    game_id SERIAL PRIMARY KEY,
    game_type_id INT NOT NULL,
    FOREIGN KEY (game_type_id) REFERENCES game_types(game_type_id),
    amount_of_players SMALLINT NOT NULL,
    flop CHAR(6),
    turn CHAR(2),
    river CHAR(2),
    final_pot DECIMAL(7,2) NOT NULL
);

-- CREATE TABLE: players_games
CREATE TABLE players_games (
    game_id INT NOT NULL,
    player_id VARCHAR(100) NOT NULL,
    starting_stack DECIMAL(7,2) NOT NULL,
    position CHAR(2) NOT NULL,
    hand CHAR(4) NOT NULL DEFAULT '????',
    winnings DECIMAL(7,2) NOT NULL DEFAULT 0,
    PRIMARY KEY (game_id, player_id)
    -- Optionally add a FK:
    -- FOREIGN KEY (game_id) REFERENCES games(game_id)
);        

-- CREATE TABLE: actions
CREATE TABLE actions (
    game_id INT NOT NULL,
    action_id INT NOT NULL,
    position CHAR(2) NOT NULL,
    action_type VARCHAR(5) NOT NULL,
    round SMALLINT NOT NULL,
    amount DECIMAL(10,2),
    pot_size DECIMAL(7,2),
    PRIMARY KEY (game_id, action_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- CREATE TABLE: players_static
CREATE TABLE players_static (
    player_id VARCHAR(100) PRIMARY KEY NOT NULL,
    total_winnings DECIMAL(15,2) DEFAULT 0,
    hands_played INT DEFAULT 1 NOT NULL
);

-- SEED game_types
INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (6, 0.10, 0.25, '$', 25);
INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (9, 0.10, 0.25, '$', 25);

INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (6, 0.25, 0.50, '$', 50);
INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (9, 0.25, 0.50, '$', 50);

INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (6, 0.50, 1, '$', 100);
INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (9, 0.50, 1, '$', 100);

INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (6, 1, 2, '$', 200);
INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (9, 1, 2, '$', 200);

INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (6, 2, 4, '$', 400);
INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (9, 2, 4, '$', 400);

INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (6, 3, 6, '$', 600);
INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (9, 3, 6, '$', 600);

INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (6, 5, 10, '$', 1000);
INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (9, 5, 10, '$', 1000);
