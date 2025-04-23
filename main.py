import os
from typing import List, Union
from dataclasses import dataclass, field

import sqlparse
from jinja2 import Environment, FileSystemLoader, Template
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection

@dataclass
class ModelConfig:
    unique: List[str] = field(default_factory=list)
    update: List[str] = field(default_factory=list)
    monitor: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        return bool(self.unique and self.update and self.monitor)

class ConfigTracker:
    def __init__(self):
        self.config_data = ModelConfig()
        self.called = False

    def set_config(
        self,
        unique: Union[str, List[str]],
        update: Union[str, List[str]],
        monitor: Union[str, List[str]]
    ) -> str:
        if self.called:
            raise RuntimeError("config() has already been called once in this template.")
        self.called = True
        self.config_data.unique = self._normalize_list(unique)
        self.config_data.update = self._normalize_list(update)
        self.config_data.monitor = self._normalize_list(monitor)
        return "-- config set"

    @staticmethod
    def _normalize_list(val: Union[str, List[str], None]) -> List[str]:
        if isinstance(val, str):
            return [val]
        return val or []

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise EnvironmentError("Missing required environment variable: DATABASE_URL")

def execute_model_sql(conn: Connection, table_name: str, select_sql: str, config: ModelConfig) -> None:
    drop_sql = f"DROP TABLE IF EXISTS `{table_name}`"
    print(f"Dropping table if exists:\n{drop_sql}\n")
    conn.execute(text(drop_sql))

    create_sql = f"CREATE TABLE `{table_name}` AS\n{select_sql}"
    print(f"Creating table:\n{create_sql}\n")
    conn.execute(text(create_sql))

    if config.unique:
        constraint_name = f"{table_name}_uniq"
        cols = ', '.join(f"`{col}`" for col in config.unique)
        alter_sql = f"ALTER TABLE `{table_name}` ADD CONSTRAINT `{constraint_name}` UNIQUE ({cols})"
        print(f"Adding unique constraint:\n{alter_sql}\n")
        conn.execute(text(alter_sql))

    conn.commit()

def create_realtime_procedure(conn: Connection, model: str, procedure_sql: str) -> None:
    print(f"Creating stored procedure for model: {model}")
    for statement in sqlparse.split(procedure_sql):
        stmt = statement.strip()
        if stmt:
            print(f"Executing statement:\n{stmt}\n")
            conn.execute(text(stmt))

def generate_realtime_procedure_sql(model: str, config: ModelConfig) -> str:
    proc_name = f"realtime_update_{model}"
    all_fields = config.unique + config.update
    param_defs = ",\n    ".join(f"IN p_{col} TEXT" for col in all_fields)
    col_names = ", ".join(f"`{col}`" for col in all_fields)
    col_values = ", ".join(f"p_{col}" for col in all_fields)
    update_clause = ", ".join(f"`{col}` = VALUES(`{col}`)" for col in config.update)

    return f"""
    DROP PROCEDURE IF EXISTS `{proc_name}`;
    CREATE PROCEDURE `{proc_name}`(
        {param_defs}
    )
    BEGIN
        INSERT INTO `{model}` ({col_names})
        VALUES ({col_values})
        ON DUPLICATE KEY UPDATE {update_clause};
    END;
    """

def main() -> None:
    engine: Engine = create_engine(DATABASE_URL)

    env: Environment = Environment(
        loader=FileSystemLoader('models/'),
        trim_blocks=True,
        lstrip_blocks=True
    )

    realtime_enabled = True  # global flag or toggle per model if needed

    with engine.begin() as conn:
        for filename in os.listdir('models/'):
            if filename.endswith('.sql'):
                model: str = os.path.splitext(filename)[0]
                tracker = ConfigTracker()
                template: Template = env.get_template(filename)

                # First render (realtime=False) to create the table
                context = {
                    "config": tracker.set_config,
                    "realtime": False,
                }
                rendered_sql = template.render(**context)
                config_data = tracker.config_data

                if not config_data.is_valid():
                    raise ValueError(f"Model '{model}' must call config() with non-empty 'unique', 'update', and 'monitor'.")

                print(f"-- Config for model: {model} --")
                print(f"Unique: {config_data.unique}")
                print(f"Update: {config_data.update}")
                print(f"Monitor: {config_data.monitor}\n")

                execute_model_sql(conn, model, rendered_sql, config_data)

                # Second render for realtime SQL (e.g. WHERE clause, duplicate update logic)
                if realtime_enabled:
                    context_realtime = {
                        "config": tracker.set_config,
                        "realtime": True,
                        "unique": config_data.unique,
                        "update": config_data.update,
                        "monitor": config_data.monitor,
                    }
                    rendered_proc_sql = generate_realtime_procedure_sql(model, config_data)
                    create_realtime_procedure(conn, model, rendered_proc_sql)

if __name__ == "__main__":
    main()
