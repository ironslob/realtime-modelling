{{
    config(
        unique = 'user_id',
        update = [
            'user_name',
            'total_count'
        ],
        monitor = {
            ref("users_animals_species_counts"): { "user_id": "user_id" },
        }
    )
}}
SELECT 
    user_id,
    user_name,
    COALESCE(SUM(count), 0) AS total_count,
    CURRENT_TIMESTAMP AS created_at,
    CURRENT_TIMESTAMP AS updated_at
FROM {{ ref('users_animals_species_counts') }}
GROUP BY user_id, user_name
