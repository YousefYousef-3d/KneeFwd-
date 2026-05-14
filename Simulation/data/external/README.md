# External reference data

Optional healthy-person step-count reference data can be placed in:

```text
data/external/healthy_steps_reference.csv
```

Expected columns are flexible, but recommended columns are:

```text
person_id,date,age,sex,steps
```

If this CSV is empty, missing, or contains no rows, the simulator creates a synthetic fallback healthy reference internally at runtime and records that choice in `outputs/data/healthy_reference_summary.csv`.
