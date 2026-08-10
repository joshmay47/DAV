"""Assits in interfacing with the intensity module."""
from .model import predict as predict_intensity
from .fitting import fit as fit_intensity

__all__ = ["predict_intensity", "fit_intensity"]
