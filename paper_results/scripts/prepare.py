"""Prepare the fixed patient folds."""

from pathlib import Path
import argparse
import contextlib
import antalya_cohort
import bwh_cohort
import json
from sklearn.model_selection import GroupKFold
import hashlib
import urllib.request


SOURCES = {
    "bwh/cycle-train-data-cleaned.csv": dict(
        url="https://ndownloader.figshare.com/files/29126475",
        sha256="aa8c3e84f032a125d8b96b993b5b6aab603b426b41b09f254e124c1f3e247f34",
        doi="10.6084/m9.figshare.15153525.v1",
    ),
    "bwh/cycle-test-data-cleaned.csv": dict(
        url="https://ndownloader.figshare.com/files/29126478",
        sha256="bae045a5dc433c74e5e6a647e77ef08f858c82046d848652f9f2734cfacbc50c",
        doi="10.6084/m9.figshare.15153525.v1",
    ),
    "antalya/electronic_data_supplementary_material.xlsx": dict(
        url="https://data.mendeley.com/public-files/datasets/xy2t2cxm8g/files/0f9a4bf7-db02-484c-b82d-e501faa06a2e/file_downloaded",
        sha256="2ec6f6cd932de7e27886f99a8e81141532794a16dd259a31f6ded6257e5cf35e",
        doi="10.17632/xy2t2cxm8g.1",
    ),
}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fetch(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name, entry in SOURCES.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            request = urllib.request.Request(
                entry["url"], headers={"User-Agent": "SC-IVF-reproducibility/1.0"}
            )
            temporary = target.with_suffix(target.suffix + ".partial")
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as stream,
            ):
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            assert sha(temporary) == entry["sha256"], (
                "Downloaded source differs",
                name,
            )
            temporary.replace(target)
        assert sha(target) == entry["sha256"], ("Local source differs", name)
    report = dict(status="passed", license="CC BY 4.0", files=SOURCES)
    (destination / "SOURCES.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"stage": "raw_sources_verified", "files": len(SOURCES)}), flush=True
    )


def prepare(raw, destination):
    raw, destination = (Path(raw).resolve(), Path(destination).resolve())
    for name, entry in SOURCES.items():
        assert sha(raw / name) == entry["sha256"], name
    receipt = destination / "DONE.json"
    if receipt.exists():
        for name, digest in json.loads(receipt.read_text())["files"].items():
            assert sha(destination / name) == digest, name
        print(json.dumps({"stage": "preparation_complete"}))
        return
    destination.mkdir(parents=True, exist_ok=True)
    bwh = destination / "bwh"
    bwh.mkdir(exist_ok=True)
    prep = bwh_cohort
    prep.PROJECT = bwh
    prep.SOURCE = raw / "bwh"
    prep.PREPARED = bwh / "data_full_pathway"
    with (
        (bwh / "source_preparation.log").open("w") as stream,
        contextlib.redirect_stdout(stream),
    ):
        prep.prepare_source()
    splitter = bwh_cohort
    splitter.PROJECT = bwh
    splitter.PREPARED = prep.PREPARED
    splitter.PARTITIONS = bwh / "prediction_restart"
    splitter.RESERVED = splitter.PARTITIONS / "sealed_test"
    with (
        (bwh / "split_preparation.log").open("w") as stream,
        contextlib.redirect_stdout(stream),
    ):
        splitter.split_development()
    ant = destination / "antalya"
    ant.mkdir(exist_ok=True)
    workbook = raw / "antalya/electronic_data_supplementary_material.xlsx"
    cleaned = antalya_cohort
    cohort, _ = cleaned.load_antalya(workbook)
    clean_path = ant / "antalya_external_clean_cohort.csv"
    cohort.to_csv(clean_path, index=False)
    data = antalya_cohort
    assert sha(clean_path) == data.SOURCE_SHA256, (
        "Raw-workbook cleaning differs from the original source"
    )
    pool, source_receipt = data._load_selected_pool(clean_path, workbook)
    assert source_receipt["holdout_values_selected"] is False
    splits = GroupKFold(n_splits=4, shuffle=True, random_state=data.FOLD_SEED).split(
        pool, groups=pool.Patient_Code
    )
    for fold, (ti, vi) in enumerate(splits):
        dest = ant / f"fold_{fold}"
        dest.mkdir(exist_ok=True)
        for name, indices in [("training", ti), ("validation", vi)]:
            pool.iloc[indices].reset_index(drop=True).to_parquet(
                dest / (name + ".parquet"), index=False
            )
    outputs = [
        destination / "bwh/prediction_restart" / name
        for name in [
            "development_parents.csv",
            "development_cycles.csv",
            "development_folds.csv",
        ]
    ]
    outputs += [
        ant / f"fold_{fold}" / (name + ".parquet")
        for fold in range(4)
        for name in ["training", "validation"]
    ]
    receipt.write_text(
        json.dumps(
            dict(
                status="complete",
                files={
                    str(path.relative_to(destination)): sha(path) for path in outputs
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps({"stage": "preparation_complete", "files": len(outputs)}), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    prepare(a.raw, a.output)
