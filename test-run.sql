\dt

SELECT * FROM animals;
SELECT * FROM species;
SELECT * FROM users;

SELECT * FROM users_animals_counts;
SELECT * FROM users_animals_counts_history;
SELECT * FROM users_animals_species_counts;

INSERT INTO animals (user_id, species_id, count) VALUES (3, 2, 5);

SELECT * FROM users_animals_counts;
SELECT * FROM users_animals_counts_history;
SELECT * FROM users_animals_species_counts;

UPDATE animals SET count = 0 WHERE species_id = 1;

SELECT * FROM users_animals_counts;
SELECT * FROM users_animals_counts_history;
SELECT * FROM users_animals_species_counts;
