from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.egoflow.config import load_config
from src.egoflow.pipeline import run_pipeline


def parse_phases(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="EgoFlow egocentric video auto-annotation pipeline")
    parser.add_argument("input_path", nargs="?", help="Input egocentric video path")
    parser.add_argument("--input", help="Input egocentric video path")
    parser.add_argument("--output", help="Optional copy target for dataset.json")
    parser.add_argument("--phases", help="Comma-separated phase list, e.g. 1,2,3")
    parser.add_argument("--resume", action="store_true", help="Skip phases with existing outputs")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI viewer")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        config = load_config(args.config)
        uvicorn.run("api.server:app", host=config["api"]["host"], port=int(config["api"]["port"]), reload=False)
        return

    input_path = args.input or args.input_path
    if not input_path:
        parser.error("--input is required unless --serve is set")

    video_uid = run_pipeline(input_path, parse_phases(args.phases), resume=args.resume, config_path=args.config)
    if args.output:
        import shutil

        dataset_path = Path(load_config(args.config)["paths"]["output_root"]) / video_uid / "dataset.json"
        shutil.copyfile(dataset_path, args.output)
    print(f"EgoFlow complete: output/{video_uid}/dataset.json")


if __name__ == "__main__":
    main()
