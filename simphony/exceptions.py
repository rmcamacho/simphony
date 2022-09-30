# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)


class SimphonyException(Exception):
    """Base class for all Simphony exceptions."""
    

class SimphonyError(SimphonyException):
    """Base class for all Simphony errors."""


class ShapeMismatchError(SimphonyError):
    """Raised when a shape mismatch occurs."""
