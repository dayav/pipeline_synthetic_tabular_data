#!/usr/bin/env python3
"""Utility to load Optuna studies pickled with older (pre-v4) TPE parameters.

Optuna 4 removed the ``consider_prior`` field from ``_ParzenEstimatorParameters``.
Pickles created with Optuna 3 therefore fail to unpickle on 4.x with
``TypeError: _ParzenEstimatorParameters.__new__() takes 7 positional arguments but 8 were given``.
This loader temporarily patches the namedtuple to drop the deprecated argument,
loads the study, restores the original constructor, and optionally re-saves the
study in the current format.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Callable

from optuna.samplers._tpe import parzen_estimator


def _patch_parzen_estimator_constructor() -> Callable[[], None]:
    """Allow unpickling Optuna 3 pickles under Optuna 4 by ignoring consider_prior."""
    params_cls = parzen_estimator._ParzenEstimatorParameters
    original_new = params_cls.__new__
    expected_len = len(params_cls._fields)

    def compat_new(cls, *args, **kwargs):
        if len(args) == expected_len + 1:
            # Drop deprecated consider_prior argument from Optuna 3 pickles.
            args = args[1:]
        return tuple.__new__(cls, args)

    params_cls.__new__ = staticmethod(compat_new)

    def restore() -> None:
        params_cls.__new__ = original_new

    return restore


def load_study(path: str) -> object:
    """Load a pickled Optuna study with backward compatibility applied."""
    restore = _patch_parzen_estimator_constructor()
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    finally:
        restore()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load an Optuna study pickle written with Optuna 3.x and re-save in the current format."
    )
    parser.add_argument("pickle_path", help="Path to the legacy Optuna study pickle.")
    parser.add_argument(
        "--out",
        help="Optional path to write the study back out using the current Optuna version.",
    )

    args = parser.parse_args()
    study = load_study(args.pickle_path)
    print(f"Loaded study with {len(getattr(study, 'trials', []))} trials from {args.pickle_path}.")

    if args.out:
        out_path = Path(args.out)
        with out_path.open("wb") as fh:
            pickle.dump(study, fh)
        print(f"Re-saved study to {out_path}")


if __name__ == "__main__":
    main()
