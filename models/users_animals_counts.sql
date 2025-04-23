{{
    config(
        unique = 'user_id',
        update = [
            'user_name',
            'total_count'
        ],
        monitor = {
            "users": { "user_id": "id" },
            ref("users_animals_species_counts"): { "user_id": "user_id" },
        }
    )
}}
SELECT 
    u.id AS user_id,
    u.name AS user_name,
    COALESCE(SUM(count), 0) AS total_count,
    CURRENT_TIMESTAMP AS created_at,
    CURRENT_TIMESTAMP AS updated_at
FROM users u
LEFT JOIN {{ ref('users_animals_species_counts') }} uasc ON (uasc.user_id = u.id)
GROUP BY u.id, u.name
