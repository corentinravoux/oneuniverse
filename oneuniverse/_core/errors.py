"""Library exception hierarchy."""


class OneuniverseError(Exception):
    """Base class of every oneuniverse library error."""


class FormatError(OneuniverseError):
    """On-disk layout or file format is invalid or unsupported."""


class SchemaError(OneuniverseError):
    """Data does not conform to the declared schema."""


class NotFoundError(OneuniverseError):
    """A requested dataset, column or entry does not exist."""


class ValidationError(OneuniverseError):
    """Input values fail validation."""
