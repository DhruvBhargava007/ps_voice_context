"""Class to handle errors from the Brain API."""

import requests
from aiohttp import ClientResponse


class BrainApiError(Exception):
    """Raised when a call to the Brain API fails."""

    def __init__(self, *, response: requests.Response | ClientResponse, extra_info: str) -> None:
        """Initialize the error."""
        if isinstance(response, requests.Response):
            self.status_code = response.status_code
        else:
            self.status_code = response.status
        self.reason = response.reason
        self.extra_info = extra_info

    def __str__(self) -> str:
        return f"Call to Brain API failed: {self.status_code} - {self.reason} - {self.extra_info}"


class BrainApiClientSideError(Exception):
    """Raised when a call to the Brain API fails because of a bad request."""

    def __init__(self, *, response: requests.Response | ClientResponse, extra_info: str) -> None:
        """Initialize the error."""
        if isinstance(response, requests.Response):
            self.status_code = response.status_code
        else:
            self.status_code = response.status
        self.reason = response.reason
        self.extra_info = extra_info

    def __str__(self) -> str:
        return f"{self.status_code} - {self.reason} - {self.extra_info}"
