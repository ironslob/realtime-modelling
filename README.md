## Pruning

Model config accepst "prune" as a boolean which defaults to true, but can be set to false. This triggers removal of records that no longer match source data, e.g. during UPDATE or DELETE.

## Incremental models

Set the unique key to a timestamp to ensure that all records are kept.
Be sure to set prune=False in the model config.
