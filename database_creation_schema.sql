DROP DATABASE pokerhands_db;
-- Creates the DB if it doesn't exist
CREATE DATABASE IF NOT EXISTS POKERHANDS_DB;

USE POKERHANDS_DB;

CREATE TABLE game_types (
	game_type_id INT PRIMARY KEY AUTO_INCREMENT,
    seat_count TINYINT NOT NULL,
    sb_amount DECIMAL(5,2) NOT NULL,
    bb_amount DECIMAL(5,2) NOT NULL,
    currency VARCHAR(4) NOT NULL DEFAULT '$',
    variant INT NOT NULL
    );
    
CREATE TABLE games (
	game_id INT PRIMARY KEY AUTO_INCREMENT,
    game_type_id INT,
    FOREIGN KEY (game_type_id) REFERENCES game_types(game_type_id),
	amount_of_players TINYINT NOT NULL,
    flop CHAR(6),
    turn CHAR(2),
    river CHAR(2),
    final_pot DECIMAL(7, 2) NOT NULL
    );
    
CREATE TABLE players_games (
	game_id INT NOT NULL,
    player_id VARCHAR(100) NOT NULL,
    starting_stack DECIMAL(7,2) NOT NULL,
    position CHAR(2) NOT NULL,
    hand CHAR(4) NOT NULL DEFAULT '????',
    winnings DECIMAL(7, 2),
    PRIMARY KEY (game_id, player_id)
    );
    
CREATE TABLE actions (
	game_id INT NOT NULL,
    action_id INT NOT NULL,
    position CHAR(2) NOT NULL, 
    action_type VARCHAR(5) NOT NULL,
    round TINYINT NOT NULL,
    amount DECIMAL(10,2),
    pot_size DECIMAL(7,2),
    PRIMARY KEY (game_id, action_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id)
    );
    
    
CREATE TABLE players_static (
	player_id VARCHAR(100) PRIMARY KEY NOT NULL,
	total_winnings DECIMAL(15, 2) DEFAULT 0,
	hands_played INT DEFAULT 1 NOT NULL
);


INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (6, 0.10, 0.25, '$', 25);

INSERT INTO game_types (seat_count, sb_amount, bb_amount, currency, variant)
VALUES (9, 0.10, 0.25, '$', 25);