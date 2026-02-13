"""Command-line-interface to run SHARP model.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import click

from . import extract_depths, predict, predict_4dgs, render


@click.group()
def main_cli():
    """Run inference for SHARP model."""
    pass


main_cli.add_command(predict.predict_cli, "predict")
main_cli.add_command(render.render_cli, "render")
main_cli.add_command(extract_depths.extract_depths_cli, "extract-depths")
main_cli.add_command(predict_4dgs.predict_4dgs_cli, "predict-4dgs")
