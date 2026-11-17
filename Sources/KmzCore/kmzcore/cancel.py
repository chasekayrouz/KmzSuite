"""Cancellation signal shared by long file operations."""


class Cancelled(Exception):
    """The user stopped an indexing or extraction job."""
