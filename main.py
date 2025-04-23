import os
import logging
import click
from typing import List, Dict, Set, Union, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection


@dataclass
class ModelConfig:
    unique: List[str] = field(default_factory=list)
    update: List[str] = field(default_factory=list)
    monitor: Dict[str, Dict[str, str]] = field(default_factory=dict)
    prune: bool = True
    prune_on_delete: bool = True
    prune_on_update: bool = True

    def is_valid(self) -> bool:
        return bool(self.unique and self.update and self.monitor)


class ConfigTracker:
    def __init__(self):
        self.config_data = ModelConfig()

    def set_config(
        self,
        unique: Union[str, List[str]],
        update: Union[str, List[str]],
        monitor: Union[Dict[str, Dict[str, str]], None],
        prune: Optional[bool] = True,
        prune_on_delete: Optional[bool] = True,
        prune_on_update: Optional[bool] = True,
    ) -> str:
        self.config_data.unique = self._normalize_list(unique)
        self.config_data.update = self._normalize_list(update)
        self.config_data.monitor = monitor or {}
        self.config_data.prune = prune
        self.config_data.prune_on_delete = prune_on_delete
        self.config_data.prune_on_update = prune_on_update
        return "-- config set"

    @staticmethod
    def _normalize_list(val: Union[str, List[str], None]) -> List[str]:
        if isinstance(val, str):
            return [val]
        return val or []


# Other helper functions like extract_dependencies, topological_sort,
# build_realtime_where_clause, run_sql, execute_model_sql,
# generate_realtime_procedure_sql, generate_realtime_prune_procedure_sql,
# create_realtime_procedure, create_prune_procedure, get_transitive_dependencies
# remain unchanged


def generate_trigger_sql(
    *,
    model: str,
    source_table: str,
    event: str,
    param_map: Dict[str, str],
    all_unique_keys: List[str],
    config: ModelConfig,
) -> List[str]:
    trigger_name = f"trigger_{source_table}_{model}_{event.lower()}"
    timing = "AFTER"
    ref_row = "NEW" if event in ("INSERT", "UPDATE") else "OLD"

    params = []
    for key in all_unique_keys:
        if key in param_map:
            params.append(f"{ref_row}.{param_map[key]}")
        else:
            params.append("NULL")
    param_str = ", ".join(params)

    sqls = [f"DROP TRIGGER IF EXISTS `{trigger_name}`;"]

    call_lines = [f"CALL `realtime_update_{model}`({param_str});"]

    if config.prune:
        if event == "DELETE" and config.prune_on_delete:
            call_lines.append(f"CALL `realtime_prune_{model}`({param_str});")
        elif event == "UPDATE" and config.prune_on_update:
            call_lines.append(f"CALL `realtime_prune_{model}`({param_str});")

    body = "\n        ".join(call_lines)

    sqls.append(f"""
    CREATE TRIGGER `{trigger_name}`
    {timing} {event} ON `{source_table}`
    FOR EACH ROW
    BEGIN
        {body}
    END;
    """)

    return sqls


@click.command()
@click.option("--debug", is_flag=True, help="Enable debug logging.")
@click.option("--dry-run", is_flag=True, help="Build everything but don’t execute SQL.")
@click.option(
    "--model",
    "model_filter",
    type=str,
    default=None,
    help="Only build this model and its dependencies.",
)
def main(debug: bool, dry_run: bool, model_filter: Optional[str]):
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

    model_sources = {
        os.path.splitext(f)[0]: open(os.path.join("models", f)).read()
        for f in os.listdir("models")
        if f.endswith(".sql")
    }

    logging.info("🔍 Extracting model dependencies...")
    dependencies = {
        model: extract_dependencies(env=env, source=source)
        for model, source in model_sources.items()
    }

    execution_order = topological_sort(dependencies)

    if model_filter:
        if model_filter not in model_sources:
            raise ValueError(f"Model '{model_filter}' not found.")
        keep = get_transitive_dependencies(model_filter, dependencies) | {model_filter}
        execution_order = [m for m in execution_order if m in keep]

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
                realtime_where_clause="",
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

            where_clause = build_realtime_where_clause(config_data.unique)
            update_proc_sql = generate_realtime_procedure_sql(
                model=model,
                config=config_data,
                raw_sql=rendered_sql,
                where_clause=where_clause,
            )
            create_realtime_procedure(
                conn=conn,
                model=model,
                sql=update_proc_sql,
                dry_run=dry_run,
                debug=debug,
            )

            if config_data.prune:
                prune_proc_sql = generate_realtime_prune_procedure_sql(
                    model=model,
                    config=config_data,
                    raw_sql=rendered_sql,
                )
                create_prune_procedure(
                    conn=conn,
                    model=model,
                    sql=prune_proc_sql,
                    dry_run=dry_run,
                    debug=debug,
                )

            for source_table, param_map in config_data.monitor.items():
                for event in ["INSERT", "UPDATE", "DELETE"]:
                    for sql in generate_trigger_sql(
                        model=model,
                        source_table=source_table,
                        event=event,
                        param_map=param_map,
                        all_unique_keys=config_data.unique,
                        config=config_data,
                    ):
                        run_sql(conn=conn, sql=sql, dry_run=dry_run, debug=debug)


if __name__ == "__main__":
    main()
