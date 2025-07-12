# Intro

A proof of concept for DBT style modelling in realtime using triggers and
stored procedures. All the models can be found in `models/`. Supports MySQL 8+ and PostgreSQL.

It was fun to build.

Do not use this in production.

# Quickstart

## Database

Using docker-compose will start up a MySQL instance *and* a PostgreSQL
instance, so you can test against either.

```
➜  realtime-modelling git:(main) ✗ docker-compose up
[+] Running 3/3
 ✔ Network realtime-modelling_default  Created                                                                      0.0s
 ✔ Container postgres                  Created                                                                      0.1s
 ✔ Container mysql                     Created                                                                      0.1s
```

It will also bootstrap them both through `bootstrap-mysql.sql` and `bootstrap-postgres.sql` respectively. Credentials are in `docker-compose.yml`.

MySQL:

```
mysql> show tables;
+------------------------------+
| Tables_in_mydb               |
+------------------------------+
| animals                      |
| species                      |
| users                        |
+------------------------------+
3 rows in set (0.005 sec)

mysql>
```

PostgreSQL:

```
mydb=# \dt
                   List of relations
 Schema |             Name             | Type  | Owner
--------+------------------------------+-------+--------
 public | animals                      | table | myuser
 public | species                      | table | myuser
 public | users                        | table | myuser
(3 rows)

mydb=#
```

## Building the models

We expect `DATABASE_URL` to be set and we'll use that for the storage. Whether it's MySQL or PostgreSQL will be detected from the SQLAlchemy URL schema. Options can be found with `--help`:

```
➜  realtime-modelling git:(main) ✗ poetry run python main.py --help
Usage: main.py [OPTIONS]

Options:
  --debug       Enable debug logging.
  --dry-run     Build everything but don't execute SQL.
  --model TEXT  Only build this model and its dependencies.
  --help        Show this message and exit.
➜  realtime-modelling git:(main) ✗
```

Simple example with `--dry-run`:

```
➜  realtime-modelling git:(main) ✗ DATABASE_URL="postgresql+psycopg2://myuser:mypassword@127.0.0.1/mydb" poetry run python main.py --dry-run

🔍 Extracting model dependencies...
🚀 Building model: users_animals_species_counts
🚀 Building model: users_animals_counts
🚀 Building model: users_animals_counts_history
➜  realtime-modelling git:(main) ✗
```

## Testing the models

All the models will have been built and populated, then you can just query them as normal.

Test run script in `test-run.sql`, run it against PostgreSQL like this:

```
➜  realtime-modelling git:(main) ✗ PGPASSWORD=mypassword psql -Umyuser -h 127.0.0.1 -d mydb -a < test-run.sql
```

# Deploying

Don't do it. It's not ready.

# Options

## Pruning

Model config accepst "prune" as a boolean which defaults to true, but can be
set to false. This triggers removal of records that no longer match source
data, e.g. during UPDATE or DELETE.

## Incremental models

Set the unique key to a timestamp to ensure that all records are kept.
Be sure to set prune=False in the model config.
