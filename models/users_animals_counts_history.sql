{{
    config(
        unique = [
            'user_id',
            'updated_at'
        ],
        update = [
            'user_name',
            'total_count'
        ],
        monitor = {
            ref("users_animals_counts"): { "user_id": "user_id" },
        }
    )
}}
SELECT * FROM {{ ref('users_animals_counts') }}
