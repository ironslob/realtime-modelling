{{
    config(
        unique = 'user_id',
        update = [
            'user_name',
            'total_count'
        ],
        monitor = {
            "users": { "user_id": "id" },
            "animals": { "user_id": "user_id" }
        }
    )
}}
-- how to specify unique index
-- all fields must be aliased
-- how to specify which tables are monitored for changes?
-- specify which actions are monitored - insert, update, delete
SELECT 
    u.id AS user_id,
    u.name AS user_name,
    COALESCE(SUM(a.count), 0) AS total_count,
    CURRENT_TIMESTAMP AS created_at,
    CURRENT_TIMESTAMP AS updated_at
FROM users u
LEFT JOIN animals a ON a.user_id = u.id
GROUP BY u.id, u.name
