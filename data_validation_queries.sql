-- Checks for missing values in the games table
SELECT * FROM games
WHERE 
    game_id IS NULL OR 
    game_type_id IS NULL OR 
    amount_of_players IS NULL OR
    final_pot IS NULL;

-- Checks for missing values in the players_games table
SELECT * FROM players_games
WHERE 
    game_id IS NULL OR 
    player_id IS NULL OR 
    starting_stack IS NULL OR
    position IS NULL OR
    hand IS NULL OR
    winnings IS NULL;

-- Checks for missing values in the actions table
SELECT * FROM actions
WHERE 
    game_id IS NULL OR 
    action_id IS NULL OR 
    position IS NULL OR
    action_type IS NULL OR
    round IS NULL OR
    pot_size IS NULL;
    
-- Checks for missing values in the players_static table
SELECT * FROM players_static
WHERE 
    player_id IS NULL OR 
    total_winnings IS NULL OR 
    hands_played IS NULL;
    
-- Checks for game_id's without any actions
SELECT g.game_id, g.game_type_id, a.*
FROM games AS g
LEFT JOIN actions AS a ON g.game_id = a.game_id
WHERE a.game_id IS NULL;

-- Checks for any actions that don't correspond to a game_id
SELECT a.game_id
FROM actions AS a
LEFT JOIN games AS g ON a.game_id = g.game_id
WHERE g.game_id IS NULL;

-- Checks for game_id's without any players in the game
SELECT g.game_id, g.amount_of_players, pg.*
FROM games AS g
LEFT JOIN players_games AS pg ON g.game_id = pg.game_id
WHERE pg.game_id IS NULL
ORDER BY g.game_id DESC;

-- Checks for players_games rows without a corresponding game_id
SELECT pg.game_id
FROM players_games AS pg
LEFT JOIN games AS g ON pg.game_id = g.game_id
WHERE g.game_id IS NULL;

-- Check for Incorrect Data Values
SELECT * FROM games WHERE amount_of_players > 9 OR amount_of_players < 2;
SELECT * FROM actions WHERE action_type NOT IN ('cc', 'cbr', 'f', 'sm');
SELECT * FROM actions WHERE amount < 0; 
SELECT * FROM players_games WHERE starting_stack <= 0 OR starting_stack >= 10000;
SELECT * FROM players_games WHERE position NOT LIKE 'p%'; 
SELECT * FROM players_games WHERE position !~ '^p[1-9]$';

-- Check for Duplicate rows in games table
SELECT game_type_id, amount_of_players, flop, turn, river, final_pot, COUNT(*)
FROM games
WHERE flop IS NOT NULL
GROUP BY game_type_id, amount_of_players, flop, turn, river, final_pot
HAVING COUNT(*) > 1;

-- Checks for invalid game_type_id's
SELECT * FROM game_types
WHERE game_type_id > 14
