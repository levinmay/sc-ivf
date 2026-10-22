"""End-to-end relational interface using the same HMA scaffold for all schemas."""

import random
import numpy as np
from .engine import SharedIVF


class IVFGenerator:
    def __init__(self, schema):
        self.schema = schema

    def fit(self, patients, cycles):
        from sdv.metadata import Metadata
        from sdv.multi_table import HMASynthesizer

        s = self.schema
        if not patients[s.patient].is_unique:
            raise ValueError("One parent row per patient required")
        if not set(cycles[s.patient]) <= set(patients[s.patient]):
            raise ValueError("Orphan cycles")
        overlap = (set(patients) & set(cycles)) - {s.patient}
        if overlap:
            raise ValueError("Patient fields must appear in the parent table only")
        self.parent_columns = list(patients)
        self.cycle_columns = list(cycles)
        self.training_patients = len(patients)
        tables = {"patients": patients.copy(), "cycles": cycles.copy()}
        metadata = Metadata.detect_from_dataframes(tables, infer_keys=None)
        for table, frame in tables.items():
            for column in frame:
                kind = (
                    "id"
                    if column in [s.patient, s.cycle]
                    else "categorical"
                    if column in s.categorical
                    else "numerical"
                )
                metadata.update_column(column, table_name=table, sdtype=kind)
        metadata.set_primary_key(s.patient, table_name="patients")
        metadata.set_primary_key(s.cycle, table_name="cycles")
        metadata.add_relationship(
            parent_table_name="patients",
            child_table_name="cycles",
            parent_primary_key=s.patient,
            child_foreign_key=s.patient,
        )
        metadata.validate_data(tables)
        self.scaffold = HMASynthesizer(metadata, verbose=False)
        self.scaffold.fit(tables)
        self.engine = SharedIVF(s).fit(
            cycles.merge(patients, on=s.patient, validate="many_to_one")
        )
        return self

    def sample(self, n_patients, seed):
        if not isinstance(n_patients, int) or n_patients < 1:
            raise ValueError("Positive integer patient count required")
        random.seed(seed)
        np.random.seed(seed)
        self.scaffold.reset_sampling()
        # SDV accepts a scale, so require the exact requested parent cardinality.
        tables = self.scaffold.sample(scale=n_patients / self.training_patients)
        s = self.schema
        if len(tables["patients"]) != n_patients:
            raise RuntimeError("HMA returned a different parent count")
        mapping = {
            value: f"S{seed}_{i}"
            for i, value in enumerate(tables["patients"][s.patient])
        }
        tables["patients"][s.patient] = tables["patients"][s.patient].map(mapping)
        tables["cycles"][s.patient] = tables["cycles"][s.patient].map(mapping)
        tables["cycles"][s.cycle] = [
            f"C{seed}_{i}" for i in range(len(tables["cycles"]))
        ]
        full = tables["cycles"].merge(
            tables["patients"], on=s.patient, validate="many_to_one"
        )
        generated = self.engine.generate(full, seed)
        return tables["patients"][self.parent_columns], generated[self.cycle_columns]
