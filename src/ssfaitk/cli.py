from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console

from .models.effort import EffortClassifier
from .models.gear import GearPredictor
from .models.vessel import VesselTypePredictor

console = Console()


def _read_df(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet"}:
        return pd.read_parquet(p)
    elif p.suffix.lower() in {".csv"}:
        return pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported input format: {p.suffix}")


def _write_df(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".parquet"}:
        df.to_parquet(p, index=False)
    elif p.suffix.lower() in {".csv"}:
        df.to_csv(p, index=False)
    else:
        raise ValueError(f"Unsupported output format: {p.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="ssfaitk", description="SSF AI Toolkit CLI")
    sub = parser.add_subparsers(dest="cmd")

    # Effort
    p_eff = sub.add_parser("effort", help="Fishing effort classification")
    p_eff_sub = p_eff.add_subparsers(dest="eff_cmd")
    p_eff_pred = p_eff_sub.add_parser("predict", help="Predict fishing vs non-fishing")
    p_eff_pred.add_argument(
        "--input", required=True, help="Parquet/CSV with track points or segments"
    )
    p_eff_pred.add_argument("--output", required=True, help="Write predictions to Parquet/CSV")
    p_eff_pred.add_argument("--model", required=False, help="Path to model artifact (optional)")

    # Gear
    p_gear = sub.add_parser("gear", help="Gear type prediction")
    p_gear_sub = p_gear.add_subparsers(dest="gear_cmd")
    p_gear_pred = p_gear_sub.add_parser("predict", help="Predict gear type")
    p_gear_pred.add_argument("--input", required=True)
    p_gear_pred.add_argument("--output", required=True)
    p_gear_pred.add_argument("--model", required=False)

    # Vessel
    p_vessel = sub.add_parser("vessel", help="Vessel type prediction")
    p_vessel_sub = p_vessel.add_subparsers(dest="vessel_cmd")
    p_vessel_pred = p_vessel_sub.add_parser("predict", help="Predict vessel type")
    p_vessel_pred.add_argument("--input", required=True)
    p_vessel_pred.add_argument("--output", required=True)
    p_vessel_pred.add_argument("--model", required=False)

    args = parser.parse_args()
    if args.cmd == "effort" and args.eff_cmd == "predict":
        df = _read_df(args.input)
        model = EffortClassifier.load(args.model) if args.model else EffortClassifier.load_default()
        out = model.predict_df(df)
        _write_df(out, args.output)
        console.print(f"[green]Effort predictions -> {args.output}[/green]")
    elif args.cmd == "gear" and args.gear_cmd == "predict":
        df = _read_df(args.input)
        model = GearPredictor.load(args.model) if args.model else GearPredictor.load_default()
        out = model.predict_df(df)
        _write_df(out, args.output)
        console.print(f"[green]Gear predictions -> {args.output}[/green]")
    elif args.cmd == "vessel" and args.vessel_cmd == "predict":
        df = _read_df(args.input)
        model = (
            VesselTypePredictor.load(args.model)
            if args.model
            else VesselTypePredictor.load_default()
        )
        out = model.predict_df(df)
        _write_df(out, args.output)
        console.print(f"[green]Vessel predictions -> {args.output}[/green]")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
