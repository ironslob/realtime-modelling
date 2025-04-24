import os
import logging
import click
from typing import List, Dict, Set, Union, Optional, Type
from dataclasses import dataclass, field
from collections import defaultdict, deque
from urllib.parse import urlparse
from abc import ABC, abstractmethod

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


def extract_dependencies(*, env: Environment, source: str) -> Set[str]:
    deps: Set[str] = set()

    def ref(name: str) -> str:
        logging.debug(f"[ref] captured: {name}")
        deps.add(name)
        return f"__REF__{name}__"

    tracker = ConfigTracker()
    env.from_string(source).render(
        config=tracker.set_config, ref=ref, realtime=False, realtime_where_clause=""
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
        raise RuntimeError("Cyclic dependency detected")

    return result


class DatabaseAdapter(ABC):
    @classmethod
    @abstractmethod
    def can_handle(cls, database_url: str) -> bool:
        """Check if this adapter can handle the given database URL."""
        pass

    @abstractmethod
    def quote_identifier(self, identifier: str) -> str:
        """Quote an identifier according to the database's syntax."""
        pass

    @abstractmethod
    def drop_table_if_exists(self, table_name: str) -> str:
        """Generate SQL to drop a table if it exists."""
        pass

    @abstractmethod
    def create_table_as(self, table_name: str, sql: str) -> str:
        """Generate SQL to create a table from a SELECT statement."""
        pass

    @abstractmethod
    def add_unique_constraint(
        self, table_name: str, constraint_name: str, columns: List[str]
    ) -> str:
        """Generate SQL to add a unique constraint."""
        pass

    @abstractmethod
    def upsert_clause(self, table_name: str, update_columns: List[str]) -> str:
        """Generate SQL for upsert operations."""
        pass

    @abstractmethod
    def create_procedure(self, name: str, params: List[str], body: str) -> str:
        """Generate SQL to create a stored procedure/function."""
        pass

    @abstractmethod
    def drop_procedure(self, name: str) -> str:
        """Generate SQL to drop a stored procedure/function."""
        pass

    @abstractmethod
    def create_trigger(
        self, name: str, table: str, timing: str, event: str, body: str
    ) -> str:
        """Generate SQL to create a trigger."""
        pass

    @abstractmethod
    def drop_trigger(self, name: str) -> str:
        """Generate SQL to drop a trigger."""
        pass

    @abstractmethod
    def generate_trigger_body(
        self,
        model: str,
        param_str: str,
        config: ModelConfig,
        event: str,
    ) -> str:
        """Generate the body of a trigger function."""
        pass


class MySQLAdapter(DatabaseAdapter):
    @classmethod
    def can_handle(cls, database_url: str) -> bool:
        return urlparse(database_url).scheme.startswith("mysql")

    def quote_identifier(self, identifier: str) -> str:
        return f"`{identifier}`"

    def drop_table_if_exists(self, table_name: str) -> str:
        return f"DROP TABLE IF EXISTS {self.quote_identifier(table_name)}"

    def create_table_as(self, table_name: str, sql: str) -> str:
        return f"CREATE TABLE {self.quote_identifier(table_name)} AS\n{sql}"

    def add_unique_constraint(
        self, table_name: str, constraint_name: str, columns: List[str]
    ) -> str:
        quoted_cols = ", ".join(self.quote_identifier(col) for col in columns)
        return f"ALTER TABLE {self.quote_identifier(table_name)} ADD CONSTRAINT {self.quote_identifier(constraint_name)} UNIQUE ({quoted_cols})"

    def upsert_clause(self, table_name: str, update_columns: List[str]) -> str:
        quoted_cols = ", ".join(
            f"{self.quote_identifier(col)} = VALUES({self.quote_identifier(col)})"
            for col in update_columns
        )
        return f"ON DUPLICATE KEY UPDATE {quoted_cols}"

    def create_procedure(self, name: str, params: List[str], body: str) -> str:
        param_defs = ",\n    ".join(f"IN {param} TEXT" for param in params)
        return f"""
        CREATE PROCEDURE {self.quote_identifier(name)}(
            {param_defs}
        )
        BEGIN
            {body}
        END;
        """

    def drop_procedure(self, name: str) -> str:
        return f"DROP PROCEDURE IF EXISTS {self.quote_identifier(name)}"

    def create_trigger(
        self, name: str, table: str, timing: str, event: str, body: str
    ) -> str:
        return f"""
        CREATE TRIGGER {self.quote_identifier(name)}
        {timing} {event} ON {self.quote_identifier(table)}
        FOR EACH ROW
        BEGIN
            {body}
        END;
        """

    def drop_trigger(self, name: str) -> str:
        return f"DROP TRIGGER IF EXISTS {self.quote_identifier(name)}"

    def generate_trigger_body(
        self,
        model: str,
        param_str: str,
        config: ModelConfig,
        event: str,
    ) -> str:
        call_lines = [
            f"CALL {self.quote_identifier(f'realtime_update_{model}')}({param_str});"
        ]
        if config.prune:
            if event == "DELETE" and config.prune_on_delete:
                call_lines.append(
                    f"CALL {self.quote_identifier(f'realtime_prune_{model}')}({param_str});"
                )
            elif event == "UPDATE" and config.prune_on_update:
                call_lines.append(
                    f"CALL {self.quote_identifier(f'realtime_prune_{model}')}({param_str});"
                )
        return "\n        ".join(call_lines)


class PostgreSQLAdapter(DatabaseAdapter):
    @classmethod
    def can_handle(cls, database_url: str) -> bool:
        return urlparse(database_url).scheme.startswith("postgresql")

    def quote_identifier(self, identifier: str) -> str:
        return f'"{identifier}"'

    def drop_table_if_exists(self, table_name: str) -> str:
        return f"DROP TABLE IF EXISTS {self.quote_identifier(table_name)} CASCADE"

    def create_table_as(self, table_name: str, sql: str) -> str:
        return f"CREATE TABLE {self.quote_identifier(table_name)} AS\n{sql}"

    def add_unique_constraint(
        self, table_name: str, constraint_name: str, columns: List[str]
    ) -> str:
        quoted_cols = ", ".join(self.quote_identifier(col) for col in columns)
        return f"ALTER TABLE {self.quote_identifier(table_name)} ADD CONSTRAINT {self.quote_identifier(constraint_name)} UNIQUE ({quoted_cols})"

    def upsert_clause(self, table_name: str, update_columns: List[str]) -> str:
        # For PostgreSQL, we need to specify the conflict columns
        # We'll use the unique constraint name that was created earlier
        constraint_name = f"{table_name}_uniq"
        quoted_cols = ", ".join(
            f"{self.quote_identifier(col)} = EXCLUDED.{self.quote_identifier(col)}"
            for col in update_columns
        )
        return f"ON CONFLICT ON CONSTRAINT {self.quote_identifier(constraint_name)} DO UPDATE SET {quoted_cols}"

    def get_column_types(self, conn: Connection, table_name: str) -> Dict[str, str]:
        """Get the column types for a table from PostgreSQL system catalogs."""
        query = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = :table_name
        """
        result = conn.execute(text(query), {"table_name": table_name})
        return {row[0]: row[1] for row in result}

    def create_procedure(
        self, name: str, params: List[str], body: str, conn: Optional[Connection] = None
    ) -> str:
        # For PostgreSQL, we'll use the actual column types if available
        param_defs = []
        if conn is not None:
            # Extract the table name from the procedure name (e.g., realtime_update_users -> users)
            table_name = name.split("_")[2]
            column_types = self.get_column_types(conn, table_name)

            for param in params:
                # Extract the column name from the parameter (remove p_ prefix)
                col_name = param[2:] if param.startswith("p_") else param
                if col_name in column_types:
                    param_defs.append(f"{param} {column_types[col_name]}")
                else:
                    param_defs.append(f"{param} anyelement")
        else:
            # Fallback to anyelement if we don't have connection info
            param_defs = [f"{param} anyelement" for param in params]

        param_defs_str = ", ".join(param_defs)
        return f"""
        CREATE OR REPLACE FUNCTION {self.quote_identifier(name)}(
            {param_defs_str}
        ) RETURNS void AS $$
        BEGIN
            {body}
        END;
        $$ LANGUAGE plpgsql;
        """

    def drop_procedure(self, name: str) -> str:
        return f"DROP FUNCTION IF EXISTS {self.quote_identifier(name)} CASCADE"

    def create_trigger(
        self, name: str, table: str, timing: str, event: str, body: str
    ) -> str:
        # First create the trigger function
        function_name = f"{name}_fn"
        function_sql = f"""
        CREATE OR REPLACE FUNCTION {self.quote_identifier(function_name)}()
        RETURNS trigger AS $$
        BEGIN
            {body}
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        """

        # Then create the trigger that uses the function
        trigger_sql = f"""
        CREATE OR REPLACE TRIGGER {self.quote_identifier(name)}
        {timing} {event} ON {self.quote_identifier(table)}
        FOR EACH ROW
        EXECUTE FUNCTION {self.quote_identifier(function_name)}();
        """

        return function_sql + "\n" + trigger_sql

    def drop_trigger(self, name: str) -> str:
        # Drop both the trigger and its associated function
        return f"""
        DROP TRIGGER IF EXISTS {self.quote_identifier(name)} ON {self.quote_identifier(name.split('_')[1])} CASCADE;
        DROP FUNCTION IF EXISTS {self.quote_identifier(name + '_fn')}() CASCADE;
        """

    def generate_trigger_body(
        self,
        model: str,
        param_str: str,
        config: ModelConfig,
        event: str,
    ) -> str:
        call_lines = [
            f"PERFORM {self.quote_identifier(f'realtime_update_{model}')}({param_str});"
        ]
        if config.prune:
            if event == "DELETE" and config.prune_on_delete:
                call_lines.append(
                    f"PERFORM {self.quote_identifier(f'realtime_prune_{model}')}({param_str});"
                )
            elif event == "UPDATE" and config.prune_on_update:
                call_lines.append(
                    f"PERFORM {self.quote_identifier(f'realtime_prune_{model}')}({param_str});"
                )
        return "\n        ".join(call_lines)


def build_realtime_where_clause(
    unique_keys: List[str],
    prefix: str = "p_",
    db_adapter: Optional[DatabaseAdapter] = None,
) -> str:
    # For all databases, use the same simple behavior
    return " AND ".join(
        f"({prefix}{key} IS NULL OR {key} = {prefix}{key})" for key in unique_keys
    )


def run_sql(*, conn: Connection, sql: str, dry_run: bool, debug: bool) -> None:
    if debug:
        logging.debug(f"Executing SQL:\n{sql.strip()}")
    if not dry_run:
        try:
            result = conn.execute(text(sql.strip()))
            conn.commit()  # Explicitly commit the transaction
            if debug:
                logging.debug(
                    f"SQL execution completed. Rows affected: {result.rowcount}"
                )
        except Exception as e:
            conn.rollback()  # Rollback on error
            logging.error(f"Error executing SQL: {str(e)}")
            raise


def get_database_adapter(database_url: str) -> DatabaseAdapter:
    """Factory function to get the appropriate database adapter."""
    adapters: List[Type[DatabaseAdapter]] = [
        MySQLAdapter,
        PostgreSQLAdapter,
    ]

    for adapter_class in adapters:
        if adapter_class.can_handle(database_url):
            return adapter_class()

    raise ValueError(f"No adapter found for database URL: {database_url}")


def execute_model_sql(
    *,
    conn: Connection,
    model: str,
    sql: str,
    config: ModelConfig,
    dry_run: bool,
    debug: bool,
    db_adapter: DatabaseAdapter,
):
    run_sql(
        conn=conn,
        sql=db_adapter.drop_table_if_exists(model),
        dry_run=dry_run,
        debug=debug,
    )
    run_sql(
        conn=conn,
        sql=db_adapter.create_table_as(model, sql),
        dry_run=dry_run,
        debug=debug,
    )

    if config.unique:
        cols = config.unique
        constraint_name = f"{model}_uniq"
        run_sql(
            conn=conn,
            sql=db_adapter.add_unique_constraint(model, constraint_name, cols),
            dry_run=dry_run,
            debug=debug,
        )


def generate_realtime_procedure_sql(
    *,
    model: str,
    config: ModelConfig,
    raw_sql: str,
    where_clause: str,
    db_adapter: DatabaseAdapter,
    conn: Optional[Connection] = None,
) -> str:
    param_defs = [f"p_{col}" for col in config.unique]
    update_clause = db_adapter.upsert_clause(model, config.update)

    return db_adapter.create_procedure(
        f"realtime_update_{model}",
        param_defs,
        f"""
        INSERT INTO {db_adapter.quote_identifier(model)}
        WITH derived AS (
            {raw_sql.strip().rstrip(';')}
        )
        SELECT * FROM derived
        WHERE {where_clause}
        {update_clause};
        """,
        conn=conn,
    )


def generate_realtime_prune_procedure_sql(
    *,
    model: str,
    config: ModelConfig,
    raw_sql: str,
    db_adapter: DatabaseAdapter,
    conn: Optional[Connection] = None,
) -> str:
    param_defs = [f"p_{col}" for col in config.unique]
    where_clause = " AND ".join(
        f"(p_{col} IS NULL OR {col} = p_{col})" for col in config.unique
    )
    key_tuple = ", ".join(config.unique)

    return db_adapter.create_procedure(
        f"realtime_prune_{model}",
        param_defs,
        f"""
        DELETE FROM {db_adapter.quote_identifier(model)}
        WHERE {where_clause}
        AND ({key_tuple}) NOT IN (
            SELECT {key_tuple} FROM (
                {raw_sql.strip().rstrip(';')}
            ) AS derived
        );
        """,
        conn=conn,
    )


def generate_trigger_sql(
    *,
    model: str,
    source_table: str,
    event: str,
    param_map: Dict[str, str],
    all_unique_keys: List[str],
    config: ModelConfig,
    db_adapter: DatabaseAdapter,
) -> List[str]:
    trigger_name = f"trigger_{source_table}_{model}_{event.lower()}"
    timing = "AFTER"
    ref_row = "NEW" if event in ("INSERT", "UPDATE") else "OLD"

    params = []
    for key in all_unique_keys:
        if key in param_map:
            params.append(f"{ref_row}.{param_map[key]}")
        else:
            # Let PostgreSQL handle the type casting automatically
            params.append("NULL")
    param_str = ", ".join(params)

    sqls = [db_adapter.drop_trigger(trigger_name)]

    body = db_adapter.generate_trigger_body(model, param_str, config, event)

    sqls.append(
        db_adapter.create_trigger(trigger_name, source_table, timing, event, body)
    )

    return sqls


def create_realtime_procedure(
    *,
    conn: Connection,
    model: str,
    sql: str,
    dry_run: bool,
    debug: bool,
    db_adapter: DatabaseAdapter,
):
    run_sql(
        conn=conn,
        sql=db_adapter.drop_procedure(f"realtime_update_{model}"),
        dry_run=dry_run,
        debug=debug,
    )
    run_sql(conn=conn, sql=sql, dry_run=dry_run, debug=debug)


def create_prune_procedure(
    *,
    conn: Connection,
    model: str,
    sql: str,
    dry_run: bool,
    debug: bool,
    db_adapter: DatabaseAdapter,
):
    run_sql(
        conn=conn,
        sql=db_adapter.drop_procedure(f"realtime_prune_{model}"),
        dry_run=dry_run,
        debug=debug,
    )
    run_sql(conn=conn, sql=sql, dry_run=dry_run, debug=debug)


def get_transitive_dependencies(
    model: str, dependencies: Dict[str, Set[str]]
) -> Set[str]:
    visited = set()

    def visit(m: str):
        if m not in visited:
            visited.add(m)
            for dep in dependencies.get(m, set()):
                visit(dep)

    visit(model)
    return visited


@click.command()
@click.option("--debug", is_flag=True, help="Enable debug logging.")
@click.option("--dry-run", is_flag=True, help="Build everything but don't execute SQL.")
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

    db_adapter = get_database_adapter(DATABASE_URL)
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
                db_adapter=db_adapter,
            )

            where_clause = build_realtime_where_clause(
                config_data.unique, db_adapter=db_adapter
            )
            update_proc_sql = generate_realtime_procedure_sql(
                model=model,
                config=config_data,
                raw_sql=rendered_sql,
                where_clause=where_clause,
                db_adapter=db_adapter,
                conn=conn,
            )
            create_realtime_procedure(
                conn=conn,
                model=model,
                sql=update_proc_sql,
                dry_run=dry_run,
                debug=debug,
                db_adapter=db_adapter,
            )

            if config_data.prune:
                prune_proc_sql = generate_realtime_prune_procedure_sql(
                    model=model,
                    config=config_data,
                    raw_sql=rendered_sql,
                    db_adapter=db_adapter,
                    conn=conn,
                )
                create_prune_procedure(
                    conn=conn,
                    model=model,
                    sql=prune_proc_sql,
                    dry_run=dry_run,
                    debug=debug,
                    db_adapter=db_adapter,
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
                        db_adapter=db_adapter,
                    ):
                        run_sql(conn=conn, sql=sql, dry_run=dry_run, debug=debug)


if __name__ == "__main__":
    main()
