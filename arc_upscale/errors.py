"""Errors that arc-upscale reports to the user."""


class UserError(Exception):
    """A problem the user can fix. Reported as one line, without a traceback."""
