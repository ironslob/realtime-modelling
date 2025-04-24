CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE species (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL UNIQUE,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE animals (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  species_id INTEGER NOT NULL,
  count INTEGER NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (species_id) REFERENCES species(id)
);

-- Insert base data BEFORE derived table creation
INSERT INTO users (name) VALUES ('Matt'), ('Dmitry'), ('Seb');
INSERT INTO species (name) VALUES ('Dog'), ('Cat'), ('Parrot');

-- Matt (id 1) has 2 dogs
INSERT INTO animals (user_id, species_id, count) VALUES (1, 1, 2);
-- Dmitry (id 2) has 1 cat and 3 parrots
INSERT INTO animals (user_id, species_id, count) VALUES (2, 2, 1);
INSERT INTO animals (user_id, species_id, count) VALUES (2, 3, 3);
-- Seb (id 3) has no animals (yet) 