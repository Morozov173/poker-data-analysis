-- Most frequent game_types
SELECT 
	gt.game_type_id, 
	gt.variant, 
	gt.seat_count, 
	COUNT(g.game_id) AS hands_of_game_type_id
FROM game_types AS gt
INNER JOIN games AS g
ON g.game_type_id = gt.game_type_id
GROUP BY gt.game_type_id
ORDER BY hands_of_game_type_id DESC , game_type_id DESC


-- Players with the highest winnings
SELECT * FROM players_static
ORDER BY total_winnings DESC

-- players that played the most hands
SELECT * FROM players_static
ORDER BY hands_played DESC

-- how many players won money in total and how many lost
SELECT 
    COUNT(*) FILTER (WHERE total_winnings < 0) AS losing_players,
    COUNT(*) FILTER (WHERE total_winnings > 0) AS winning_players,
    COUNT(*) AS total_players
FROM players_static;

-- 10 biggest pots played
SELECT g.*, gt.variant FROM games AS g
LEFT JOIN game_types AS gt
ON gt.game_type_id = g.game_type_id
ORDER BY final_pot DESC
LIMIT 10



SELECT pg.position, COUNT(*) AS num_hands, SUM(pg.winnings) AS total_winnings
FROM players_games AS pg
WHERE EXISTS (
    SELECT 1
    FROM games AS g
    INNER JOIN game_types AS gt ON g.game_type_id = gt.game_type_id
    WHERE gt.seat_count = 6 AND gt.variant = 400 AND g.game_id = pg.game_id
)
GROUP BY pg.position
ORDER BY total_winnings DESC;


-- distrubiton of winnings according to position in 6Max
SELECT pg.position, 
       COUNT(*) AS hands_played, 
       SUM(pg.winnings) AS total_winnings, 
       AVG(pg.winnings) AS avg_winnings_per_hand_played
FROM players_games AS pg
WHERE EXISTS (
    SELECT 1
    FROM games AS g
    INNER JOIN game_types AS gt ON g.game_type_id = gt.game_type_id
    WHERE gt.seat_count = 6 AND gt.variant = 1000 AND g.game_id = pg.game_id
)
GROUP BY pg.position
ORDER BY avg_winnings_per_hand_played DESC;

