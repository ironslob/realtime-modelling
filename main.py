import os
import logging
import click
from typing import List, Dict, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection

# --- Model Configuration Tracking ---

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

# --- Dependency Extraction ---

def extract_dependencies(*, env: Environment, source: str) -> Set[str]:
    deps: Set[str] = set()

    def ref(name: str) -> str:
        logging.debug(f"[ref] captured: {name}")
        deps.add(name)
        return f"__REF__{name}__"

    tracker = ConfigTracker()
    env.from_string(source).render(
        config=tracker.set_config,
        ref=ref,
        realtime=False,
        realtime_where_clause=""
    )
    return deps

def topological_sort(dependencies: Dict[str, Set[str]]) -> List[str]:
    in_degree = {model: 0 for model in dependencies}
    reverse_deps = defaultdict(set)

    for model, deps in dependencies.items():
        for dep in deps:
            if dep not in in_degree:
                raise ValueError(f"Model '{model}' depends on undefined model '{dep}'")
            in_degree[model] += 1
            reverse_deps[dep].add(model)

    queue = deque([m for m, deg in in_degree.items() if deg == 0])
    result = []

    while queue:
        model = queue.popleft()
        result.append(model)
        for dependent in reverse_deps[model]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(dependencies):
        logging.error("Dependency graph incomplete:")
        logging.debug(f"Full dependency graph: {dependencies}")
        logging.error(f"Resolved: {result}")
        logging.error(f"Remaining: {[m for m in dependencies if m not in result]}")
        raise RuntimeError("Cyclic dependency detected")

    return result

# --- SQL Helpers ---

def build_realtime_where_clause(unique_keys: List[str], prefix: str = "p_") -> str:
    return ' AND '.join(f"{key} = {prefix}{key}" for key in unique_keys)

def run_sql(*, conn: Connection, sql: str, dry_run: bool, debug: bool) -> None:
    if debug:
        logging.debug(f"Executing SQL:\n{sql.strip()}")
    if not dry_run:
        conn.execute(text(sql.strip()))

# --- Model Builders ---

def execute_model_sql(*, conn: Connection, model: str, sql: str, config: ModelConfig, dry_run: bool, debug: bool):
    run_sql(conn=conn, sql=f"DROP TABLE IF EXISTS `{model}`", dry_run=dry_run, debug=debug)
    run_sql(conn=conn, sql=f"CREATE TABLE `{model}` AS\n{sql}", dry_run=dry_run, debug=debug)

    if config.unique:
        cols = ', '.join(f"`{col}`" for col in config.unique)
        constraint_name = f"{model}_uniq"
        run_sql(conn=conn, sql=f"ALTER TABLE `{model}` ADD CONSTRAINT `{constraint_name}` UNIQUE ({cols})", dry_run=dry_run, debug=debug)

def generate_realtime_procedure_sql(*, model: str, config: ModelConfig, raw_sql: str, where_clause: str) -> str:
    param_defs = ",\n    ".join(f"IN p_{col} TEXT" for col in config.unique)
    update_clause = ", ".join(f"`{col}` = VALUES(`{col}`)" for col in config.update)

    return f"""
    CREATE PROCEDURE `realtime_update_{model}`(
        {param_defs}
    )
    BEGIN
        WITH derived AS (
            {raw_sql.strip().rstrip(';')}
        )
        INSERT INTO `{model}`
        SELECT * FROM derived
        WHERE {where_clause}
        ON DUPLICATE KEY UPDATE {update_clause};
    END;
    """

def create_realtime_procedure(*, conn: Connection, model: str, sql: str, dry_run: bool, debug: bool):
    run_sql(conn=conn, sql=f"DROP PROCEDURE IF EXISTS `realtime_update_{model}`;", dry_run=dry_run, debug=debug)
    run_sql(conn=conn, sql=sql, dry_run=dry_run, debug=debug)

# --- Main CLI ---

@click.command()
@click.option("--debug", is_flag=True, help="Enable debug logging.")
@click.option("--dry-run", is_flag=True, help="Build everything but don’t execute SQL.")
def main(debug: bool, dry_run: bool):
    if debug:
        logging.basicConfig(level=logging.DEBUG, format="[debug] %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise EnvironmentError("Missing DATABASE_URL")

    engine: Engine = create_engine(DATABASE_URL)
    env: Environment = Environment(
        loader=FileSystemLoader("models/"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Load model files
    model_sources = {
        os.path.splitext(f)[0]: open(os.path.join("models", f)).read()
        for f in os.listdir("models")
        if f.endswith(".sql")
    }

    # Build dependency graph
    logging.info("🔍 Extracting model dependencies...")
    dependencies = {
        model: extract_dependencies(env=env, source=source)
        for model, source in model_sources.items()
    }

    execution_order = topological_sort(dependencies)
    logging.info(f"✅ Execution order: {execution_order}")

    with engine.connect() as conn:
        for model in execution_order:
            logging.info(f"🚀 Building model: {model}")
            source = model_sources[model]
            tracker = ConfigTracker()

            rendered_sql = env.from_string(source).render(
                config=tracker.set_config,
                ref=lambda name: name,
                realtime=False,
                realtime_where_clause=""
            )

            config_data = tracker.config_data
            if not config_data.is_valid():
                raise ValueError(f"Model '{model}' is missing config values")

            execute_model_sql(
                conn=conn,
                model=model,
                sql=rendered_sql,
                config=config_data,
                dry_run=dry_run,
                debug=debug,
            )

            # Create realtime procedure
            where_clause = build_realtime_where_clause(config_data.unique)
            rendered_proc_sql = generate_realtime_procedure_sql(
                model=model,
                config=config_data,
                raw_sql=rendered_sql,
                where_clause=where_clause
            )
            create_realtime_procedure(
                conn=conn,
                model=model,
                sql=rendered_proc_sql,
                dry_run=dry_run,
                debug=debug,
            )

if __name__ == "__main__":
    main()
