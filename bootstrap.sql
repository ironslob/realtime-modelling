CREATE TABLE users (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(255) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE species (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(255) NOT NULL UNIQUE,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE animals (
  id INT PRIMARY KEY AUTO_INCREMENT,
  user_id INT NOT NULL,
  species_id INT NOT NULL,
  count INT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
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

