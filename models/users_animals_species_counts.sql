{{
    config(
        unique = [
            'user_id',
            'species_id'
        ],
        update = [
            'user_name',
            'count',
            'updated_at'
        ],
        monitor = {
            "users": { "user_id": "id" },
            "animals": { "user_id": "user_id" }
        }
    )
}}
SELECT 
    u.id AS user_id,
    u.name AS user_name,
    s.id AS species_id,
    s.name AS species_name,
    COALESCE(SUM(a.count), 0) AS count,
    CURRENT_TIMESTAMP AS created_at,
    CURRENT_TIMESTAMP AS updated_at
FROM users u
JOIN animals a ON a.user_id = u.id
JOIN species s ON s.id = a.species_id
GROUP BY u.id, s.id
