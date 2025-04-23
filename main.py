import os
import logging
import click
from typing import List, Dict, Union
from dataclasses import dataclass, field

from jinja2 import Environment, FileSystemLoader, Template, StrictUndefined
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection

@dataclass
class ModelConfig:
    unique: List[str] = field(default_factory=list)
    update: List[str] = field(default_factory=list)
    monitor: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return bool(self.unique and self.update and self.monitor)

class ConfigTracker:
    def __init__(self):
        self.config_data = ModelConfig()

    def set_config(
        self,
        unique: Union[str, List[str]],
        update: Union[str, List[str]],
        monitor: Union[Dict[str, Dict[str, str]], None]
    ) -> str:
        self.config_data.unique = self._normalize_list(unique)
        self.config_data.update = self._normalize_list(update)
        self.config_data.monitor = monitor or {}
        return "-- config set"

    @staticmethod
    def _normalize_list(val: Union[str, List[str], None]) -> List[str]:
        if isinstance(val, str):
            return [val]
        return val or []

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise EnvironmentError("Missing required environment variable: DATABASE_URL")

def build_realtime_where_clause(unique_keys: List[str], prefix: str = "p_") -> str:
    if not unique_keys:
        return ""
    conditions = [f"{key} = {prefix}{key}" for key in unique_keys]
    return ' AND '.join(conditions)

def run_sql(*, conn: Connection, sql: str, dry_run: bool, debug: bool) -> None:
    if debug:
        logging.debug(f"Executing SQL:\n{sql.strip()}")
    if not dry_run:
        conn.execute(text(sql.strip()))

def execute_model_sql(*, conn: Connection, table_name: str, select_sql: str, config: ModelConfig, dry_run: bool, debug: bool) -> None:
    run_sql(conn=conn, sql=f"DROP TABLE IF EXISTS `{table_name}`", dry_run=dry_run, debug=debug)
    run_sql(conn=conn, sql=f"CREATE TABLE `{table_name}` AS\n{select_sql}", dry_run=dry_run, debug=debug)

    if config.unique:
        constraint_name = f"{table_name}_uniq"
        cols = ', '.join(f"`{col}`" for col in config.unique)
        alter_sql = f"ALTER TABLE `{table_name}` ADD CONSTRAINT `{constraint_name}` UNIQUE ({cols})"
        run_sql(conn=conn, sql=alter_sql, dry_run=dry_run, debug=debug)

def generate_realtime_procedure_sql(*, model: str, config: ModelConfig, rendered_select_sql: str, where_clause: str) -> str:
    proc_name = f"realtime_update_{model}"
    param_defs = ",\n    ".join(f"IN p_{col} TEXT" for col in config.unique)
    update_clause = ", ".join(f"`{col}` = VALUES(`{col}`)" for col in config.update)

    # CTE wrapping and WHERE clause on derived table
    rendered_select_sql = rendered_select_sql.strip().rstrip(';')
    wrapped_sql = f"WITH derived AS (\n{rendered_select_sql}\n)\nSELECT * FROM derived WHERE {where_clause}"

    return f"""
    CREATE PROCEDURE `{proc_name}`(
        {param_defs}
    )
    BEGIN
        INSERT INTO `{model}`
        {wrapped_sql}
        ON DUPLICATE KEY UPDATE {update_clause};
    END;
    """

def create_realtime_procedure(*, conn: Connection, model: str, procedure_sql: str, dry_run: bool, debug: bool) -> None:
    proc_name = f"realtime_update_{model}"
    run_sql(conn=conn, sql=f"DROP PROCEDURE IF EXISTS `{proc_name}`;", dry_run=dry_run, debug=debug)
    run_sql(conn=conn, sql=procedure_sql, dry_run=dry_run, debug=debug)

def create_triggers(*, conn: Connection, model: str, config: ModelConfig, dry_run: bool, debug: bool) -> None:
    for table, param_map in config.monitor.items():
        for action in ["INSERT", "UPDATE", "DELETE"]:
            timing = "AFTER"
            trigger_name = f"realtime_{model}_{table}_after_{action.lower()}"
            row_ref = "NEW" if action in ["INSERT", "UPDATE"] else "OLD"

            run_sql(conn=conn, sql=f"DROP TRIGGER IF EXISTS `{trigger_name}`;", dry_run=dry_run, debug=debug)

            arg_values = []
            for param_name in config.unique:
                source_column = param_map.get(param_name)
                if not source_column:
                    raise ValueError(f"Missing mapping for unique param '{param_name}' in monitor[{table}]")
                arg_values.append(f"{row_ref}.{source_column}")

            call_proc = f"CALL realtime_update_{model}({', '.join(arg_values)});"
            trigger_sql = f"""
            CREATE TRIGGER `{trigger_name}`
            {timing} {action} ON `{table}`
            FOR EACH ROW
            BEGIN
                {call_proc}
            END;
            """
            run_sql(conn=conn, sql=trigger_sql, dry_run=dry_run, debug=debug)

@click.command()
@click.option('--debug', is_flag=True, help="Enable SQL debug logging.")
@click.option('--dry-run', is_flag=True, help="Prepare but do not execute SQL.")
def main(debug: bool, dry_run: bool) -> None:
    if debug:
        logging.basicConfig(level=logging.DEBUG, format='[DEBUG] %(message)s')

    engine: Engine = create_engine(DATABASE_URL)
    env: Environment = Environment(
        loader=FileSystemLoader('models/'),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined
    )

    with engine.connect() as conn:
        for filename in os.listdir('models/'):
            if filename.endswith('.sql'):
                model = os.path.splitext(filename)[0]
                tracker = ConfigTracker()
                template = env.get_template(filename)

                context = {
                    "config": tracker.set_config,
                    "realtime": False,
                    "realtime_where_clause": "",
                }
                rendered_sql = template.render(**context)
                config_data = tracker.config_data

                if not config_data.is_valid():
                    raise ValueError(f"Model '{model}' must call config() with 'unique', 'update', and 'monitor'.")

                execute_model_sql(
                    conn=conn,
                    table_name=model,
                    select_sql=rendered_sql,
                    config=config_data,
                    dry_run=dry_run,
                    debug=debug,
                )

                where_clause = build_realtime_where_clause(config_data.unique)
                context_realtime = {
                    "config": tracker.set_config,
                    "realtime": True,
                    "unique": config_data.unique,
                    "update": config_data.update,
                    "monitor": config_data.monitor,
                    "primary_key": config_data.unique[0] if config_data.unique else None,
                    "realtime_where_clause": where_clause,
                }
                rendered_proc_sql = template.render(**context_realtime)
                proc_sql = generate_realtime_procedure_sql(
                    model=model,
                    config=config_data,
                    rendered_select_sql=rendered_proc_sql,
                    where_clause=where_clause,
                )
                create_realtime_procedure(
                    conn=conn,
                    model=model,
                    procedure_sql=proc_sql,
                    dry_run=dry_run,
                    debug=debug,
                )
                create_triggers(
                    conn=conn,
                    model=model,
                    config=config_data,
                    dry_run=dry_run,
                    debug=debug,
                )

if __name__ == "__main__":
    main()



