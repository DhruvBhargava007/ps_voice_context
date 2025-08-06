"""Class providing access to Brain API endpoints."""

import inspect
import json
import os
from collections.abc import Generator
from datetime import datetime
from typing import Any, BinaryIO, ClassVar, Literal, Optional, Type, TypeVar, Union, cast
from uuid import UUID

import aiohttp
import requests
import sseclient
from aiohttp import ClientResponse
from pydantic import BaseModel, Json

from brain_platform_client.brain_api_schemas import (
    MODEL_TYPES,
    Agent,
    AgentResponse,
    AiModelUpdate,
    AiModelUpdateResponse,
    AiModelWithPricing,
    AiModelWithPricingData,
    BatchMetadataResponse,
    BatchMetadataResponseList,
    BatchRequest,
    BatchRetrieveStatusResponse,
    BatchStatusEnum,
    BatchStatusResponse,
    Chunk,
    ChunkCreate,
    ChunkResponseList,
    Collection,
    Conversation,
    ConversationCostResponse,
    ConversationCostResponseData,
    ConversationResponse,
    ConversationWithMessages,
    ConversationWithMessagesResponse,
    ConversationWithMessagesResponseData,
    CreateChunksResponse,
    DeleteTokenResponse,
    DeleteTranscriptionResponse,
    Document,
    DocumentPresignedUrl,
    DocumentPresignedUrlResponse,
    DocumentStatus,
    EmbeddingsResponse,
    FastBatchResponse,
    FastBatchResponseList,
    GenerateImageResponse,
    GenerateImageResponseData,
    GetAgentsResponse,
    GetCollectionResponse,
    GetCollectionsResponse,
    GetCsvResponse,
    GetDocumentResponse,
    GetTokenResponse,
    GetTokenResponseInfo,
    GetTokenResponseList,
    InvokeLlmResponse,
    InvokeLlmResponseData,
    IssueTokenInfo,
    IssueTokenResponse,
    PrivateConversationMessage,
    SearchResponse,
    SearchResponseList,
    SearchWithContextResponse,
    SearchWithContextResponseList,
    TokenUsage,
    TokenUsageResponse,
    TokenUsageResponseList,
    Tool,
    ToolUseResponse,
    ToolUseResponseData,
    TranscriptionResponse,
    TranscriptionResponseData,
    TranscriptionResponseList,
    TranscriptionWorker,
    TranscriptionWorkerResponse,
    UpdateTokenResponse,
    UserConversationCostResponse,
    UserConversationCostResponseData,
)
from brain_platform_client.brain_client_error import BrainApiClientSideError, BrainApiError
from brain_platform_client.utils import flatten_json_schema

T = TypeVar("T", bound=BaseModel)


class BrainApi:
    """Class providing access to Brain API endpoints."""

    def __init__(
        self,
        *,
        user_email: Optional[str] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        timeout: Optional[int] = None,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        """Initialize a BrainApi client.

        Args:
            user_email: (optional) The email of the user to track usage in Brain.
            api_key: (optional) Brain API key. If not provided, will look for BRAIN_API_KEY in environment variables.
            url: (optional) The URL of the Brain API. If not provided, will look for BRAIN_URL in environment variables
                or default to https://brain-platform.pattern.com.
            timeout: (optional) The timeout for requests in seconds. Default is 30 seconds.
            aiohttp_session: (optional) An aiohttp.ClientSession object to use for async requests.
        """
        self.api_key = api_key or os.environ.get("BRAIN_API_KEY")
        self.url = url or os.environ.get("BRAIN_URL") or "https://brain-platform.pattern.com"
        if not self.api_key:
            raise ValueError("api_key must be provided")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        # if user_email is provided, use it to construct a user id to track usage in Brain
        if user_email:
            self.headers["X-User-ID"] = user_email

        # set timeout from user input or fallback on default
        DEFAULT_TIMEOUT = 30  # seconds
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.aiohttp_session = aiohttp_session

    SUCCESS_CODES: ClassVar[set[int]] = {200, 201, 204}

    CLIENT_SIDE_ERROR_CODES: ClassVar[set[int]] = {400}

    _M = TypeVar("_M", bound=BaseModel)

    def parsed_response(self, response: requests.Response, model: Type[_M]) -> _M:
        """Try to get JSON from response."""

        if response.status_code not in self.SUCCESS_CODES:
            is_json = response.headers.get("content-type") == "application/json"
            if is_json:
                extra_info = response.json().get("detail", response.json())
            else:
                extra_info = response.text
            if response.status_code in self.CLIENT_SIDE_ERROR_CODES:
                raise BrainApiClientSideError(response=response, extra_info=extra_info)
            raise BrainApiError(response=response, extra_info=extra_info)

        json = response.json()
        validated = model.model_validate(json)

        return validated

    async def async_parsed_response(self, response: ClientResponse, model: Type[_M]) -> _M:
        """Try to get JSON from response with async."""

        if response.status not in self.SUCCESS_CODES:
            is_json = response.headers.get("content-type") == "application/json"
            extra_info = await response.json() if is_json else await response.text()
            raise BrainApiError(response=response, extra_info=extra_info)

        json = await response.json()
        validated = model.model_validate(json)

        return validated

    EMBEDDING_MODEL = "text-embedding-3-large"

    def v1(self, endpoint: str) -> str:
        """Construct a v1 route."""
        endpoint = endpoint.lstrip("/")
        return f"{self.url}/api/v1/{endpoint}"

    def v2(self, endpoint: str) -> str:
        """Construct a v2 route."""
        endpoint = endpoint.lstrip("/")
        return f"{self.url}/api/v2/{endpoint}"

    def v1_get(
        self,
        endpoint: str,
        *,
        params: Any = {},
        timeout: Optional[int] = None,
    ) -> requests.Response:
        """Make a GET request to a v1 route.

        Args:
            endpoint: The endpoint to call.
            params: The parameters to pass to the endpoint.
            timeout: The timeout for the request.  Overrides the client-level default timeout if provided.
        """
        return requests.get(
            self.v1(endpoint),
            params=params,
            headers=self.headers,
            timeout=timeout or self.timeout,
        )

    async def async_v1_get(
        self,
        endpoint: str,
        *,
        params: dict[str, str | int] = {},
        timeout: Optional[int] = None,
    ) -> ClientResponse:
        """Make a GET request to a v1 route.

        Args:
            endpoint: The endpoint to call.
            params: The parameters to pass to the endpoint.
            timeout: The timeout for the request.  Overrides the client-level default timeout if provided.
        """
        assert self.aiohttp_session is not None, "aiohttp_session must be provided for async requests."
        return await self.aiohttp_session.get(
            self.v1(endpoint),
            params=params,
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=timeout or self.timeout),
        )

    def v2_get(
        self,
        endpoint: str,
        *,
        params: Any = {},
        timeout: Optional[int] = None,
    ) -> requests.Response:
        """Make a GET request to a v2 route.

        Args:
            endpoint: The endpoint to call.
            params: The parameters to pass to the endpoint.
            timeout: The timeout for the request.  Overrides the client-level default timeout if provided.
        """
        return requests.get(
            self.v2(endpoint),
            params=params,
            headers=self.headers,
            timeout=timeout or self.timeout,
        )

    async def async_v2_get(
        self,
        endpoint: str,
        *,
        params: dict[str, str | int] = {},
        timeout: Optional[int] = None,
    ) -> ClientResponse:
        """Make a GET request to a v2 route.

        Args:
            endpoint: The endpoint to call.
            params: The parameters to pass to the endpoint.
            timeout: The timeout for the request.  Overrides the client-level default timeout if provided.
        """
        assert self.aiohttp_session is not None, "aiohttp_session must be provided in for async requests."
        return await self.aiohttp_session.get(
            self.v2(endpoint),
            params=params,
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=timeout or self.timeout),
        )

    def v1_post(
        self,
        endpoint: str,
        *,
        json_data: Optional[dict[str, Any]] = None,
        files: Optional[Any] = None,
        stream: bool = False,
        timeout: Optional[int] = None,
    ) -> requests.Response:
        """Make a POST request to a v1 route.

        Args:
            endpoint: The endpoint to call.
            json_data: The JSON data to pass to the endpoint.
            files: The files to pass to the endpoint.
            stream: Whether to stream the response.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided.
        """
        return requests.post(
            self.v1(endpoint),
            files=files,
            data=json_data if files is not None else None,
            json=json_data if files is None else None,
            headers=self.headers,
            timeout=timeout or self.timeout,
            stream=stream,
        )

    async def async_v1_post(
        self,
        endpoint: str,
        *,
        json_data: Optional[dict[str, Any]] = None,
        files: Optional[aiohttp.FormData] = None,
        timeout: Optional[int] = None,
    ) -> ClientResponse:
        """Make a POST request to a v1 route.

        Args:
            endpoint: The endpoint to call.
            json_data: The JSON data to pass to the endpoint.
            files: The files to pass to the endpoint.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided.
        """
        assert self.aiohttp_session is not None, "aiohttp_session must be provided in for async requests."

        # if files and json_data are both provided, add json_data to files
        if files and json_data:
            for key, value in json_data.items():
                files.add_field(key, value)
            json_data = None

        return await self.aiohttp_session.post(
            self.v1(endpoint),
            data=files if files is not None else None,
            json=json_data if files is None else None,
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=timeout or self.timeout),
        )

    def v2_post(
        self,
        endpoint: str,
        *,
        json_data: Optional[dict[str, Any]] = None,
        files: Optional[Any] = None,
        stream: bool = False,
        timeout: Optional[int] = None,
    ) -> requests.Response:
        """Make a POST request to a v2 route.

        Args:
            endpoint: The endpoint to call.
            json_data: The JSON data to pass to the endpoint.
            files: The files to pass to the endpoint.
            stream: Whether to stream the response.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided.
        """
        return requests.post(
            self.v2(endpoint),
            files=files,
            data=json_data if files is not None else None,
            json=json_data if files is None else None,
            headers=self.headers,
            timeout=timeout or self.timeout,
            stream=stream,
        )

    async def async_v2_post(
        self,
        endpoint: str,
        *,
        json_data: Optional[dict[str, Any]] = None,
        files: Optional[aiohttp.FormData] = None,
        timeout: Optional[int] = None,
    ) -> ClientResponse:
        """Make a POST request to a v2 route.

        Args:
            endpoint: The endpoint to call.
            json_data: The JSON data to pass to the endpoint.
            files: The files to pass to the endpoint.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided.
        """
        assert self.aiohttp_session is not None, "aiohttp_session must be provided in for async requests."

        # if files and json_data are both provided, add json_data to files
        if files and json_data:
            for key, value in json_data.items():
                files.add_field(key, value)
            json_data = None

        return await self.aiohttp_session.post(
            self.v2(endpoint),
            data=files if files is not None else None,
            json=json_data if files is None else None,
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=timeout or self.timeout),
        )

    def v1_patch(
        self,
        endpoint: str,
        *,
        json_data: Optional[dict[str, Any]],
        timeout: Optional[int] = None,
    ) -> requests.Response:
        """Make a PATCH request to a v1 route.

        Args:
            endpoint: The endpoint to call.
            json_data: The JSON data to pass to the endpoint.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided
        """
        return requests.patch(
            self.v1(endpoint),
            json=json_data,
            headers=self.headers,
            timeout=timeout or self.timeout,
        )

    async def async_v1_patch(
        self,
        endpoint: str,
        *,
        json_data: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> ClientResponse:
        """Make a PATCH request to a v1 route.

        Args:
            endpoint: The endpoint to call.
            json_data: The JSON data to pass to the endpoint.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided.
        """
        assert self.aiohttp_session is not None, "aiohttp_session must be provided for async requests."
        return await self.aiohttp_session.patch(
            self.v1(endpoint),
            json=json_data,
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=timeout or self.timeout),
        )

    def v2_patch(
        self,
        endpoint: str,
        *,
        json_data: Optional[dict[str, Any]],
        timeout: Optional[int] = None,
    ) -> requests.Response:
        """Make a PATCH request to a v2 route.

        Args:
            endpoint: The endpoint to call.
            json_data: The JSON data to pass to the endpoint.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided
        """
        return requests.patch(
            self.v2(endpoint),
            json=json_data,
            headers=self.headers,
            timeout=timeout or self.timeout,
        )

    async def async_v2_patch(
        self,
        endpoint: str,
        *,
        json_data: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> ClientResponse:
        """Make a PATCH request to a v2 route.

        Args:
            endpoint: The endpoint to call.
            json_data: The JSON data to pass to the endpoint.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided.
        """
        assert self.aiohttp_session is not None, "aiohttp_session must be provided for async requests."
        return await self.aiohttp_session.patch(
            self.v2(endpoint),
            json=json_data,
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=timeout or self.timeout),
        )

    def v1_delete(
        self,
        endpoint: str,
        *,
        timeout: Optional[int] = None,
    ) -> requests.Response:
        """Make a DELETE request to a v1 route.

        Args:
            endpoint: The endpoint to call.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided.
        """
        return requests.delete(
            self.v1(endpoint),
            headers=self.headers,
            timeout=timeout or self.timeout,
        )

    async def async_v1_delete(
        self,
        endpoint: str,
        *,
        timeout: Optional[int] = None,
    ) -> ClientResponse:
        """Make a DELETE request to a v1 route.

        Args:
            endpoint: The endpoint to call.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided.
        """
        assert self.aiohttp_session is not None, "aiohttp_session must be provided for async requests."
        return await self.aiohttp_session.delete(
            self.v1(endpoint),
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=timeout or self.timeout),
        )

    def v2_delete(
        self,
        endpoint: str,
        *,
        timeout: Optional[int] = None,
    ) -> requests.Response:
        """Make a DELETE request to a v2 route.

        Args:
            endpoint: The endpoint to call.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided.
        """
        return requests.delete(
            self.v2(endpoint),
            headers=self.headers,
            timeout=timeout or self.timeout,
        )

    async def async_v2_delete(
        self,
        endpoint: str,
        *,
        timeout: Optional[int] = None,
    ) -> ClientResponse:
        """Make a DELETE request to a v2 route.

        Args:
            endpoint: The endpoint to call.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided.
        """
        assert self.aiohttp_session is not None, "aiohttp_session must be provided for async requests."
        return await self.aiohttp_session.delete(
            self.v2(endpoint),
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=timeout or self.timeout),
        )

    def v1_put(
        self,
        endpoint: str,
        *,
        json_data: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> requests.Response:
        """Make a PUT request to a v1 route.

        Args:
            endpoint: The endpoint to call.
            json_data: The JSON data to pass to the endpoint.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided.
        """
        return requests.put(
            self.v1(endpoint),
            json=json_data,
            headers=self.headers,
            timeout=timeout or self.timeout,
        )

    async def async_v1_put(
        self,
        endpoint: str,
        *,
        json_data: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> ClientResponse:
        """Make a PUT request to a v1 route.

        Args:
            endpoint: The endpoint to call.
            json_data: The JSON data to pass to the endpoint.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided.
        """
        assert self.aiohttp_session is not None, "aiohttp_session must be provided for async requests."
        return await self.aiohttp_session.put(
            self.v1(endpoint),
            json=json_data,
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=timeout or self.timeout),
        )

    def v2_put(
        self,
        endpoint: str,
        *,
        json_data: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> requests.Response:
        """Make a PUT request to a v2 route.

        Args:
            endpoint: The endpoint to call.
            json_data: The JSON data to pass to the endpoint.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided.
        """
        return requests.put(
            self.v2(endpoint),
            json=json_data,
            headers=self.headers,
            timeout=timeout or self.timeout,
        )

    async def async_v2_put(
        self,
        endpoint: str,
        *,
        json_data: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> ClientResponse:
        """Make a PUT request to a v2 route.

        Args:
            endpoint: The endpoint to call.
            json_data: The JSON data to pass to the endpoint.
            timeout: The timeout for the request. Overrides the client-level default timeout if provided.
        """
        assert self.aiohttp_session is not None, "aiohttp_session must be provided for async requests."
        return await self.aiohttp_session.put(
            self.v2(endpoint),
            json=json_data,
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=timeout or self.timeout),
        )

    ########################################
    # API methods
    ########################################

    # Conversations

    def get_user_cost(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> UserConversationCostResponseData:
        """Get the cost of a user.

        Args:
            start_date: The start date of the cost to get.
            end_date: The end date of the cost to get.
            tags: A dictionary of tags to filter by.

        Returns:
            A UserConversationCostResponseData object.
        """
        response = self.v1_get(
            "/conversations/cost",
            params={"start_date": start_date, "end_date": end_date, "tags": tags},
        )
        parsed = self.parsed_response(response, UserConversationCostResponse)
        return parsed.data

    def get_conversation_cost(self, conversation_id: Union[str, UUID]) -> ConversationCostResponseData:
        """Get the cost of a conversation.

        Args:
            conversation_id: The id of the conversation to get the cost of.

        Returns:
            A ConversationCostResponseData object.

        Raises:
            Exception: If a conversation with that id does not exist for the user.
        """
        response = self.v1_get(f"/conversations/{conversation_id!s}/cost")
        parsed = self.parsed_response(response, ConversationCostResponse)
        return parsed.data

    def list_conversations(
        self,
        include_archived: Optional[bool] = False,
    ) -> list[Conversation]:
        """Get all conversations for the user.

        Returns
            A list of Conversation objects.
        """
        response = self.v1_get(
            "/conversations",
            params={"include_archived": include_archived},
        )
        parsed = self.parsed_response(response, ConversationResponse)
        # Normalize data to always be a list
        return parsed.data if isinstance(parsed.data, list) else [parsed.data]

    def get_conversation(self, conversation_id: Union[str, UUID], days_old: Optional[int] = None) -> ConversationWithMessages:
        """Get a conversation and it's messages by conversation ID.

        Args:
            conversation_id: The id of the conversation to get.
            days_old: The number of days for which to retrieve conversation history.

        Returns:
            The Conversation object.

        Raises:
            Exception: If a conversation with that id does not exist for the user.
        """
        params: dict[str, int] = {}
        if days_old:
            params["days_old"] = days_old
        response = self.v1_get(f"/conversations/{conversation_id!s}", params=params)
        parsed = self.parsed_response(response, ConversationWithMessagesResponse)
        return parsed.data

    def archive_conversation(self, conversation_id: Union[str, UUID]) -> Conversation:
        """Archive a conversation by ID.

        Args:
            conversation_id: The id of the conversation to archive.

        Returns:
             A list of Conversation objects.

        Raises:
            Exception: If a conversation with that id does not exist for the user.
        """
        response = self.v1_put(f"/conversations/{conversation_id!s}/archive", json_data={})
        parsed = self.parsed_response(response, ConversationWithMessagesResponseData)
        return parsed.data

    # LLMs

    def list_llms(
        self,
        include_archived: Optional[bool] = False,
        ai_model_type: Optional[MODEL_TYPES] = None,
    ) -> list[AiModelWithPricing]:
        """List all LLMs available to the user.

        Args:
            include_archived: Include archived LLMs. Default is False.
            ai_model_type: Filter by AI model type. If None, all types are returned.
        """
        params: dict[str, Any] = {
            "include_archived": include_archived,
            "ai_model_type": ai_model_type,
        }
        response = self.v1_get("/models", params=params)
        parsed = self.parsed_response(response, AiModelWithPricingData)
        return parsed.data

    async def async_list_llms(
        self,
        include_archived: Optional[bool] = False,
        ai_model_type: Optional[MODEL_TYPES] = None,
    ) -> list[AiModelWithPricing]:
        """List all LLMs available to the user.

        Args:
            include_archived: Include archived LLMs. Default is False.
            ai_model_type: Filter by AI model type. If None, all types are returned.
        """
        params: dict[str, str | int] = {
            "include_archived": json.dumps(include_archived),
        }
        if ai_model_type:
            params["ai_model_type"] = ai_model_type

        response = await self.async_v1_get("/models", params=params)
        parsed = await self.async_parsed_response(response, AiModelWithPricingData)
        return parsed.data

    def create_llm(self, model_data: AiModelWithPricing) -> AiModelWithPricing:
        """Create a new LLM model."""
        # Convert the model data to a dict and handle datetime serialization
        model_dict = json.loads(model_data.model_dump_json())
        if isinstance(model_dict.get("knowledge_cutoffs"), datetime):
            model_dict["knowledge_cutoffs"] = model_dict["knowledge_cutoffs"].isoformat()

        response = self.v1_post("/models/create", json_data=model_dict)
        parsed = self.parsed_response(response, AiModelUpdateResponse)
        return parsed.data

    def update_llm(self, model_id: str, model_data: AiModelUpdate) -> AiModelWithPricing:
        """Update an existing LLM model."""
        model_dict = json.loads(model_data.model_dump_json())

        response = self.v1_put(f"/models/update/{model_id}", json_data=model_dict)
        parsed = self.parsed_response(response, AiModelUpdateResponse)
        return parsed.data

    def invoke_llm(
        self,
        *,
        prompt: str,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        conversation_id: Optional[Union[str, UUID]] = None,
        collection_id: Optional[Union[str, UUID]] = None,
        summarize_conversation_name: Optional[bool] = False,
        response_format: Optional[Union[Literal["json_object", "text"], Type[T]]] = None,
        image_urls: Optional[list[str]] = None,
        image_resolution: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        private: Optional[bool] = False,
        list_of_messages: Optional[list[PrivateConversationMessage]] = None,
        enable_fallback: Optional[bool] = True,
        video_url: Optional[str] = None,
        file_data: Optional[bytes] = None,
    ) -> InvokeLlmResponseData:
        """Invoke an LLM. To stream, use invoke_llm_stream().

        Args:
            prompt (required): The user message to send to the LLM.
            model: A valid LLM model to use. See list_llms().
            system_message: A system message to supply to the LLM.
            temperature: How "creative" the LLM should be. 0.1 is very conservative, 1.0 is very creative.
            conversation_id: Use to continue an existing conversation.
            collection_id: This forces a retrieval of relevant documents. If not helpful, LLM will ignore.
            summarize_conversation_name: If True, the conversation name will be summarized.
            response_format: The format of the response. "json_object" or "text". Default is "text".
                            Must include 'json' in prompt if using "json_object". If using a BaseModel, pass the class.
                            Only specific GPT models support BaseModel.
            image_urls: A list of image urls to use with the LLM.
            image_resolution: The resolution of the image, 'high' or 'low'. Default is 'auto'.
            tags: A dictionary of tags to filter by.
            private: If True, the conversation will be private.
            list_of_messages: A list of messages to send to the LLM. when Conversation is private.


        Returns:
            An InvokeLlmResponseData object.
        """
        # Convert response_format to a JSON-serializable format if it is a subclass of BaseModel
        response_format_: dict[str, Any] = {}
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            response_format_ = response_format.model_json_schema()
            response_format_ = flatten_json_schema(response_format_)
        # Convert list_of_messages to a JSON-serializable format
        list_of_messages_ = [message.model_dump() for message in list_of_messages] if list_of_messages else None

        json_data: dict[str, Any] = {
            "prompt": prompt,
            "system_message": system_message,
            "temperature": temperature,
            "conversation_id": str(conversation_id) if conversation_id else None,
            "collection_id": str(collection_id) if collection_id else None,
            "summarize_conversation_name": summarize_conversation_name,
            "response_format": response_format_ if response_format_ else response_format,
            "image_urls": image_urls,
            "image_resolution": image_resolution,
            "tags": tags,
            "private": private,
            "list_of_messages": list_of_messages_,
            "enable_fallback": enable_fallback,
            "video_url": video_url,
            "file_data": file_data,
        }
        if model:
            # If model is provided, add it to the json_data otherwise it will use the default model from the API
            json_data["model"] = model
        response = self.v1_post("/llms/invoke", json_data=json_data, timeout=900)
        parsed = self.parsed_response(response, InvokeLlmResponse)
        return parsed.data

    def generate_batch_embeddings(
        self,
        texts: list[str],
        model: Optional[str] = None,  # intentionally setting to None to use default from API
        dimensions: Optional[int] = None,  # intentionally setting to None to use default from
    ):
        """Generate embeddings for a batch of texts."""
        json_data: dict[str, Any] = {
            "texts": texts,
            "model": model,
            "dimensions": dimensions,
        }
        response = self.v1_post("/embeddings", json_data=json_data, timeout=900)
        parsed = self.parsed_response(response, EmbeddingsResponse)
        return parsed.data

    def invoke_llm_tool_use(
        self,
        *,
        prompt: str,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        conversation_id: Optional[Union[str, UUID]] = None,
        collection_id: Optional[Union[str, UUID]] = None,
        summarize_conversation_name: Optional[bool] = False,
        response_format: Optional[Union[Literal["json_object", "text"], Type[T]]] = None,
        image_urls: Optional[list[str]] = None,
        tags: Optional[dict[str, str]] = None,
        tools: list[Tool],
        tool_choice: Optional[str] = "auto",
    ) -> ToolUseResponseData:
        """Invoke an LLM with tool calling. Currently only supports OpenAI models.

        Args:
            prompt (required): The user message to send to the LLM.
            model: A valid LLM model to use. See list_llms().
            system_message: A system message to supply to the LLM.
            temperature: How "creative" the LLM should be. 0.1 is very conservative, 1.0 is very creative.
            conversation_id: Use to continue an existing conversation.
            collection_id: This forces a retrieval of relevant documents. If not helpful, LLM will ignore.
            summarize_conversation_name: If True, the conversation name will be summarized.
            response_format: The format of the response. "json_object" or "text". Default is "text".
                            Must include 'json' in prompt if using "json_object". If using a BaseModel, pass the class.
                            Only specific GPT models support BaseModel.
            image_urls: A list of image urls to use with the LLM.
            tags: A dictionary of tags to filter by.
            tools: A list of tools to use with the LLM. Import the 'Tool' class from brain_api_schemas.
            tool_choice: "auto" = any tool or msg, "required" = any tool, "none" = msg, "tool_name" = specific tool.

        Returns:
            A ToolUseResponseData object.
        """
        # Convert response_format to a JSON-serializable format if it is a subclass of BaseModel
        response_format_: dict[str, Any] = {}
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            response_format_ = response_format.model_json_schema()

        # Convert tools to a JSON-serializable format
        tools_ = [tool.model_dump() for tool in tools] if tools else None
        json_data: dict[str, Any] = {
            "prompt": prompt,
            "system_message": system_message,
            "temperature": temperature,
            "conversation_id": str(conversation_id) if conversation_id else None,
            "collection_id": str(collection_id) if collection_id else None,
            "summarize_conversation_name": summarize_conversation_name,
            "response_format": response_format_ if response_format_ else response_format,
            "image_urls": image_urls,
            "tags": tags,
            "tools": tools_,
            "tool_choice": tool_choice,
        }
        if model:
            # If model is provided, add it to the json_data otherwise it will use the default model from the API
            json_data["model"] = model
        response = self.v1_post("/llms/invoke/tool_use", json_data=json_data, timeout=900)
        parsed = self.parsed_response(response, ToolUseResponse)
        return parsed.data

    async def async_invoke_llm(
        self,
        *,
        prompt: str,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        conversation_id: Optional[Union[str, UUID]] = None,
        collection_id: Optional[Union[str, UUID]] = None,
        summarize_conversation_name: Optional[bool] = False,
        response_format: Optional[Union[Literal["json_object", "text"], Type[T]]] = None,
        image_urls: Optional[list[str]] = None,
        image_resolution: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        private: Optional[bool] = False,
        list_of_messages: Optional[list[PrivateConversationMessage]] = None,
        enable_fallback: Optional[bool] = True,
        video_url: Optional[str] = None,
        file_data: Optional[bytes] = None,
    ) -> InvokeLlmResponseData:
        """Invoke an LLM. To stream, use invoke_llm_stream().

        Args:
            prompt (required): The user message to send to the LLM.
            model: A valid LLM model to use. See list_llms().
            system_message: A system message to supply to the LLM.
            temperature: How "creative" the LLM should be. 0.1 is very conservative, 1.0 is very creative.
            conversation_id: Use to continue an existing conversation.
            collection_id: This forces a retrieval of relevant documents. If not helpful, LLM will ignore.
            summarize_conversation_name: If True, the conversation name will be summarized.
            response_format: The format of the response. "json_object" or "text". Default is "text".
                            Must include 'json' in prompt if using "json_object". If using a BaseModel, pass the class.
                            Only specific GPT models support BaseModel.
            image_urls: A list of image urls to use with the LLM.
            image_resolution: The resolution of the image, 'high' or 'low'. Default is 'auto'.
            tags: A dictionary of tags to filter by.
            private: If True, the conversation will be private.
            list_of_messages: A list of messages to send to the LLM. when Conversation is private.


        Returns:
            An InvokeLlmResponseData object.
        """
        # Convert response_format to a JSON-serializable format if it is a subclass of BaseModel
        response_format_: dict[str, Any] = {}
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            response_format_ = response_format.model_json_schema()
            response_format_ = flatten_json_schema(response_format_)
        # Convert list_of_messages to a JSON-serializable format
        list_of_messages_ = [message.model_dump() for message in list_of_messages] if list_of_messages else None

        json_data: dict[str, Any] = {
            "prompt": prompt,
            "system_message": system_message,
            "temperature": temperature,
            "conversation_id": str(conversation_id) if conversation_id else None,
            "collection_id": str(collection_id) if collection_id else None,
            "summarize_conversation_name": summarize_conversation_name,
            "response_format": response_format_ if response_format_ else response_format,
            "image_urls": image_urls,
            "image_resolution": image_resolution,
            "tags": tags,
            "private": private,
            "list_of_messages": list_of_messages_,
            "enable_fallback": enable_fallback,
            "video_url": video_url,
            "file_data": file_data,
        }
        if model:
            # If model is provided, add it to the json_data otherwise it will use the default model from the API
            json_data["model"] = model
        response = await self.async_v1_post("/llms/invoke", json_data=json_data, timeout=900)
        parsed = await self.async_parsed_response(response, InvokeLlmResponse)
        return parsed.data

    def invoke_llm_stream(
        self,
        *,
        prompt: str,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        conversation_id: Optional[Union[str, UUID]] = None,
        collection_id: Optional[Union[str, UUID]] = None,
        summarize_conversation_name: Optional[bool] = None,
        response_format: Optional[Union[Literal["json_object", "text"], Type[T]]] = None,
        image_urls: Optional[list[str]] = None,
        image_resolution: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        private: Optional[bool] = False,
        list_of_messages: Optional[list[PrivateConversationMessage]] = None,
        enable_fallback: Optional[bool] = True,
        video_url: Optional[str] = None,
        file_data: Optional[bytes] = None,
    ) -> sseclient.SSEClient:
        """Invoke an LLM and stream the response. To not stream, use invoke_llm().

        Args:
            prompt (required): The user message to send to the LLM.
            model: A valid LLM model to use. See list_llms().
            system_message: A system message to supply to the LLM.
            temperature: How "creative" the LLM should be. 0.1 is very conservative, 1.0 is very creative.
            conversation_id: Use to continue an existing conversation.
            collection_id: This forces a retrieval of relevant documents. If not helpful, LLM will ignore.
            response_format: The format of the response. "json_object" or "text". Default is "text".
                            Must include 'json' in prompt if using "json_object". If using a BaseModel, pass the class.
                            Only specific GPT models support BaseModel.
            summarize_conversation_name: If True, the conversation name will be summarized.
            image_urls: A list of image urls to use with the LLM.
            image_resolution: The resolution of the image, 'high' or 'low'. Default is 'auto'.
            tags: A dictionary of tags to filter by.
            private: If True, the conversation will be private.
            list_of_messages: A list of messages to send to the LLM. when Conversation is private.

        Returns:
            A sseclient.SSEClient object.
        """
        # Convert response_format to a JSON-serializable format if it is a subclass of BaseModel
        response_format_: dict[str, Any] = {}
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            response_format_ = response_format.model_json_schema()
            response_format_ = flatten_json_schema(response_format_)
        # Convert list_of_messages to a JSON-serializable format
        list_of_messages_ = [message.model_dump() for message in list_of_messages] if list_of_messages else None

        json_data: dict[str, Any] = {
            "prompt": prompt,
            "system_message": system_message,
            "temperature": temperature,
            "conversation_id": str(conversation_id) if conversation_id else None,
            "collection_id": str(collection_id) if collection_id else None,
            "summarize_conversation_name": summarize_conversation_name,
            "response_format": response_format_ if response_format_ else response_format,
            "stream": True,
            "image_urls": image_urls,
            "image_resolution": image_resolution,
            "tags": tags,
            "private": private,
            "list_of_messages": list_of_messages_,
            "enable_fallback": enable_fallback,
            "video_url": video_url,
            "file_data": file_data,
        }
        if model:
            # If model is provided, add it to the json_data otherwise it will use the default model from the API
            json_data["model"] = model
        response = self.v1_post("/llms/invoke", json_data=json_data, stream=True, timeout=900)
        response = cast(Generator[Any, Any, Any], response)  # makes type checker happy
        return sseclient.SSEClient(response)

    # Image endpoints

    def generate_image(
        self,
        model: str,
        prompt: str,
        image_urls: Optional[list[str]] = None,
        conversation_id: Optional[str | UUID] = None,
        summarize_conversation_name: Optional[bool] = False,
        background: Optional[Literal["opaque", "transparent", "auto"]] = "auto",
        size: Optional[Literal["square", "landscape", "portrait", "auto"]] = "auto",
        n: Optional[int] = 1,
    ) -> GenerateImageResponse:
        """Generate an image from text.

        Args:
            model: The model to use. See list_image_models().
            prompt: The text to generate the image from.
            conversation_id: Use to continue an existing conversation.
            summarize_conversation_name: If True, the conversation name will be summarized.
        """
        json_data: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "image_urls": image_urls if image_urls else None,
            "conversation_id": str(conversation_id) if conversation_id else None,
            "summarize_conversation_name": summarize_conversation_name,
            "background": background if background else None,
            "size": size if size else None,
            "n": n if n is not None else None,
        }
        response = self.v1_post("/images/generations", json_data=json_data, timeout=900)
        parsed = self.parsed_response(response, GenerateImageResponseData)
        return parsed.data

    async def async_generate_image(
        self,
        model: str,
        prompt: str,
        image_urls: Optional[list[str]] = None,
        conversation_id: Optional[str | UUID] = None,
        summarize_conversation_name: Optional[bool] = False,
        background: Optional[Literal["opaque", "transparent", "auto"]] = "auto",
        size: Optional[Literal["square", "landscape", "portrait", "auto"]] = "auto",
        n: Optional[int] = 1,
    ) -> GenerateImageResponse:
        """Generate an image from text.

        Args:
            model: The model to use. See list_image_models().
            prompt: The text to generate the image from.
            conversation_id: Use to continue an existing conversation.
            summarize_conversation_name: If True, the conversation name will be summarized.
        """
        json_data: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "image_urls": image_urls if image_urls else None,
            "conversation_id": str(conversation_id) if conversation_id else None,
            "summarize_conversation_name": summarize_conversation_name,
            "background": background if background else None,
            "size": size if size else None,
            "n": n if n else None,
        }
        request_context_manager = await self.async_v1_post("/images/generations", json_data=json_data, timeout=900)
        async with request_context_manager as response:
            parsed = await self.async_parsed_response(response, GenerateImageResponseData)
        return parsed.data

    # Collections

    def list_collections(self, include_public_collections: bool = False) -> list[Collection]:
        """List all Collections for the user.

        Returns
            A list of Collection objects.
        """
        params: dict[str, Any] = {
            "include_public_collections": include_public_collections,
        }
        response = self.v2_get("/collections", params=params)
        parsed = self.parsed_response(response, GetCollectionsResponse)
        return parsed.data

    def create_collection(
        self,
        collection_name: str,
        embedding_model: str = EMBEDDING_MODEL,
        dimension: int = 512,
        description: Optional[str] = None,
    ) -> Collection:
        """Create a Collection.

        Args:
            collection_name: The name of the collection.
            embedding_model: The embedding model to use. Default is "text-embedding-3-large".
            dimension: The dimensions for the
            description: The description of the collection

        Returns:
            The created Collection object.
        """
        response = self.v2_post(
            "/collections",
            json_data={
                "collection_name": collection_name,
                "dimension": dimension,
                "embedding_model": embedding_model,
                "description": description,
            },
            timeout=900,
        )
        parsed = self.parsed_response(response, GetCollectionResponse)
        return parsed.data

    def add_users_to_collection(self, collection_id: str | UUID, user_emails: list[str]) -> Collection:
        """Add users to a collection.

        Args:
            collection_id: The id of the collection to add users to.
            user_emails: A list of user emails to add to the collection.

        Returns:
            The Collection object.
        """
        response = self.v2_post(
            f"/collections/{collection_id!s}/add_users",
            json_data={"user_emails": user_emails},
        )
        parsed = self.parsed_response(response, GetCollectionResponse)
        return parsed.data

    def delete_collection(self, collection_id: str | UUID) -> Collection:
        """Delete a Collection by ID.

        Args:
            collection_id: The id of the collection to delete.

        Returns:
            The Collection object.

        Raises:
            Exception: If a collection with that id does not exist for the user.
        """
        response = self.v2_delete(f"/collections/{collection_id}")
        parsed = self.parsed_response(response, GetCollectionResponse)
        return parsed.data

    async def async_delete_collection(self, collection_id: str | UUID) -> Collection:
        """Delete a Collection by ID.

        Args:
            collection_id: The id of the collection to delete.

        Returns:
            The Collection object.

        Raises:
            Exception: If a collection with that id does not exist for the user.
        """
        request_context_manager = await self.async_v2_delete(f"/collections/{collection_id!s}")
        async with request_context_manager as response:
            parsed = await self.async_parsed_response(response, GetCollectionResponse)
        return parsed.data

    # Documents

    def list_documents(self, collection_id: str | UUID) -> list[DocumentPresignedUrl]:
        """List all Documents in a Collection.

        Args:
            collection_id: The id of the collection to get documents from.

        Returns:
            A list of Document objects. Just attributes, not the content.

        Raises:
            Exception: If a collection with that id does not exist for the user.
        """
        response = self.v2_get(
            "/documents",
            params={"collection_id": str(collection_id)},
        )
        parsed = self.parsed_response(response, DocumentPresignedUrlResponse)
        return parsed.data

    def get_document(self, document_id: str | UUID) -> Document:
        """Get a Document by ID.

        Args:
            document_id: The id of the document to get.

        Returns:
            The Document object. Just attributes, not the content.

        Raises:
            Exception: If a document with that id does not exist for the user.
        """
        response = self.v2_get(f"/documents/{document_id!s}")
        parsed = self.parsed_response(response, GetDocumentResponse)
        return parsed.data

    def create_document(
        self,
        document: BinaryIO,
        collection_id: str | UUID,
        source: Optional[str] = None,
        document_id: Optional[str | UUID] = None,
    ) -> Document:
        """Create a document.

        Args:
            document_id: The id of the document to create. If None, a new id will be generated.
            document: A file object. Recommended types: TXT, SQL, JSON, RB, PY, MD, IPYNB, CSV
            collection_id: The id of the collection to add the document to.
            source: The source of the document. Often a URL or file path.

        Returns:
            The created Document object.

        Raises:
            Exception: If the file is not a valid format.
        """
        response = self.v2_post(
            "/documents",
            files=[("file", (document.name, document.read(), "application/octet-stream"))],
            json_data={"document_id": document_id, "collection_id": str(collection_id), "source": source},
            timeout=900,
        )
        parsed = self.parsed_response(response, GetDocumentResponse)
        return parsed.data

    def create_csv(self, documents: BinaryIO, collection_id: str | UUID) -> list[str]:
        """Advanced method to create documents from a CSV file.

        Args:
            documents: A CSV file.
            collection_id: The id of the collection to add the documents to.

        Returns:
            A list of document ids.

        Raises:
            Exception: If the file is not CSV or not properly formatted.
        """
        response = self.v2_post(
            "/documents/csv",
            files=[("file", (documents.name, documents.read(), "application/octet-stream"))],
            json_data={"collection_id": str(collection_id)},
            timeout=900,
        )
        parsed = self.parsed_response(response, GetCsvResponse)
        return parsed.data

    def patch_document_attributes(
        self,
        document_id: str | UUID,
        document_name: Optional[str] = None,
        source: Optional[str] = None,
        author: Optional[str] = None,
        associated_content: Optional[str] = None,
    ) -> Document:
        """Patch document attributes.

        Args:
            document_id: The id of the document to patch.
            document_name: The name of the document.
            source: The source of the document. Often a URL or file path.
            author: The human who wrote the document.
            status: Chunking/embedding status of the document.
            associated_content: Any additional metadata about the document.

        Returns:
            The patched Document object.

        Raises:
        Exception: If a document with that id does not exist
        """
        json_data = {
            "document_name": document_name,
            "source": source,
            "author": author,
            "associated_content": associated_content,
        }
        response = self.v2_patch(f"/documents/{document_id!s}", json_data=json_data)
        parsed = self.parsed_response(response, GetDocumentResponse)
        return parsed.data

    async def async_patch_document_attributes(
        self,
        document_id: str | UUID,
        document_name: Optional[str] = None,
        source: Optional[str] = None,
        author: Optional[str] = None,
        associated_content: Optional[str] = None,
    ) -> Document:
        """Patch document attributes.

        Args:
            document_id: The id of the document to patch.
            document_name: The name of the document.
            source: The source of the document. Often a URL or file path.
            author: The human who wrote the document.
            status: Chunking/embedding status of the document.
            associated_content: Any additional metadata about the document.

        Returns:
            The patched Document object.

        Raises:
        Exception: If a document with that id does not exist
        """
        json_data = {
            "document_name": document_name,
            "source": source,
            "author": author,
            "associated_content": associated_content,
        }
        request_context_manager = await self.async_v1_patch(f"/documents/{document_id!s}", json_data=json_data)
        async with request_context_manager as response:
            parsed = await self.async_parsed_response(response, GetDocumentResponse)
        return parsed.data

    def update_document_status(self, document_id: str | UUID, status: DocumentStatus) -> Document:
        """Update the status of a document in the Brain API."""
        json_data = {"status": status}
        response = self.v2_patch(
            f"/documents/{document_id!s}/status",
            json_data=json_data,
        )
        parsed = self.parsed_response(response, GetDocumentResponse)
        return parsed.data

    async def async_update_document_status(self, document_id: str | UUID, status: DocumentStatus) -> Document:
        """Update the status of a document in the Brain API."""
        json_data = {"status": status}
        request_context_manager = await self.async_v2_patch(f"/documents/{document_id!s}/status", json_data=json_data)
        async with request_context_manager as response:
            parsed = await self.async_parsed_response(response, GetDocumentResponse)
        return parsed.data

    def get_document_attributes(self, document_id: str | UUID) -> Document:
        """Get document attributes from Brain API."""
        response = self.v2_get(f"/documents/{document_id!s}")
        parsed = self.parsed_response(response, GetDocumentResponse)
        return parsed.data

    def delete_document(self, document_id: str | UUID) -> Document:
        """Delete a document from the Brain API."""
        response = self.v2_delete(f"/documents/{document_id!s}")
        parsed = self.parsed_response(response, GetDocumentResponse)
        return parsed.data

    async def async_delete_document(self, document_id: str | UUID) -> Document:
        """Delete a document from the Brain API."""
        request_context_manager = await self.async_v2_delete(f"/documents/{document_id!s}")
        async with request_context_manager as response:
            parsed = await self.async_parsed_response(response, GetDocumentResponse)
        return parsed.data

    # Search

    def search(self, content: str, collection_id: str | UUID, num_of_results: int) -> list[SearchResponse]:
        """Search for documents containing the content.

        Args:
            content: The content to search for.
            collection_id: The collection to search in.
            num_of_results: The number of results to return.

        Returns:
            A list of SearchResponse objects.
        """
        response = self.v2_post(
            "/search",
            json_data={
                "content": content,
                "collection_id": str(collection_id),
                "num_of_results": num_of_results,
            },
            timeout=900,
        )
        parsed = self.parsed_response(response, SearchResponseList)
        return parsed.data

    def search_with_context(
        self, content: str, collection_id: (str | UUID), num_of_results: int
    ) -> list[SearchWithContextResponse]:
        """Search for documents containing the content.

        Args:
            content: The content to search for.
            collection_id: The collection to search in.
            num_of_results: The number of results to return.

        Returns:
            A list of SearchWithContextResponse objects.
        """
        response = self.v2_post(
            "/search_with_context",
            json_data={
                "content": content,
                "collection_id": str(collection_id),
                "num_of_results": num_of_results,
            },
            timeout=900,
        )
        parsed = self.parsed_response(response, SearchWithContextResponseList)
        return parsed.data

    # Audio Transcription

    def create_transcription(
        self,
        *,
        audio: BinaryIO,
        timestamp_granularity: Optional[Literal["word", "segment"]] = None,
        language: Optional[str] = None,
    ) -> TranscriptionWorkerResponse:
        """Create a transcription from an audio file.

        Args:
            audio: Valid ext are .m4a, .mp3, .mp4, .mpeg, .mpga, .wav, .webm.
            timestamp_granularity: The granularity of the timestamps in the transcription [word, segment]
            language: The language of the audio file (e.g., "en", "es", "fr"). If not provided, language will be auto-detected.

        Returns:
            The created Transcription object.

        Raises:
            Exception: If the audio file is not one of the valid formats.
        """
        file = [("audio", (audio.name, audio.read(), "application/octet-stream"))]
        json_data = {"timestamp_granularity": timestamp_granularity, "language": language}

        response = self.v1_post(
            "/audio/transcriptions",
            files=file,
            json_data=json_data,
            timeout=900,
        )
        parsed = self.parsed_response(response, TranscriptionWorker)
        return parsed.data

    def get_transcription_by_id(self, transcription_id: UUID | str) -> TranscriptionResponse:
        """Get a transcription by ID.

        Args:
            transcription_id: The id of the transcription to get.

        Returns:
            The Transcription object.

        Raises:
            Exception: If a transcription with that id does not exist for the user.
        """
        response = self.v1_get(f"/audio/{transcription_id!s}")
        parsed = self.parsed_response(response, TranscriptionResponseData)
        return parsed.data

    def get_transcriptions(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> list[TranscriptionResponse]:
        """List all transcription with user_id.

        Args:
            start_date: Start of time for filter transcription.
            end_date: End of time for filter transcription.

        Returns:
            The Transcription object.

        Raises:
            Exception:
                If a start_date & end_date not matched with the standard format.
                If transcription not found
        """
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        response = self.v1_get("/audio", params=params)
        parsed = self.parsed_response(response, TranscriptionResponseList)
        return parsed.data

    def delete_transcription(self, transcription_id: UUID | str) -> str:
        """Delete a transcription by ID.

        Args:
            transcription_id: The id of the transcription to delete.

        Returns:
            The Transcription object.

        Raises:
            Exception: If a transcription with that id does not exist for the user.
        """
        response = self.v1_delete(f"/audio/{transcription_id!s}")
        parsed = self.parsed_response(response, DeleteTranscriptionResponse)
        return parsed.data

    # Agents

    def list_agents(self, include_public_agents: bool = False) -> list[Agent]:
        """List all agents associated with the user.

        Args:
            include_public_agents (bool): Flag to include public agents.

        Returns:
            A list of Agents objects.
        """
        params: dict[str, Any] = {
            "include_public_agents": include_public_agents,
        }
        response = self.v1_get("/agents/", params=params)
        parsed = self.parsed_response(response, GetAgentsResponse)
        return parsed.data

    def create_agent(self, name: str, model: str, instructions: str) -> Agent:
        """Create an agent.

        Args:
            name: The name of the agent.
            model: Any model from the list of models.
            instructions: The instructions for the agent.

        Returns:
            The created agent.

        Raises:
            Exception: If the model name is invalid.
        """
        json_data = {"name": name, "model": model, "instructions": instructions}
        response = self.v1_post("/agents", json_data=json_data)
        parsed = self.parsed_response(response, AgentResponse)
        return parsed.data

    def add_users_to_agent(self, agent_id: str | UUID, user_emails: list[str]) -> Agent:
        """Add users to an agent.

        Args:
            agent_id: The id of the agent to add users to.
            user_emails: A list of user emails to add to the agent.

        Returns:
            The Agent object.
        """
        response = self.v1_post(
            f"/agents/{agent_id!s}/add_users",
            json_data={"user_emails": user_emails},
        )
        parsed = self.parsed_response(response, AgentResponse)
        return parsed.data

    def get_agent(self, agent_id: str | UUID) -> Agent:
        """Get an agent by id.

        Args:
            agent_id: The id of the agent to get.

        Returns:
            The agent associated with that id.

        Raises:
            Exception: If an agent with that id does not exist
            or is not associated with the user.
        """
        response = self.v1_get(f"/agents/{agent_id!s}")
        parsed = self.parsed_response(response, AgentResponse)
        return parsed.data

    def delete_agent(self, agent_id: str | UUID) -> Agent:
        """Delete an agent.

        Args:
            agent_id: The id of the agent to delete.

        Returns:
            The deleted agent.

        Raises:
            Exception: If an agent with that id does not exist
            or is not associated with the user.
        """
        response = self.v1_delete(f"/agents/{agent_id!s}")
        parsed = self.parsed_response(response, AgentResponse)
        return parsed.data

    async def async_delete_agent(self, agent_id: str | UUID) -> Agent:
        """Delete an agent.

        Args:
            agent_id: The id of the agent to delete.

        Returns:
            The deleted agent.

        Raises:
            Exception: If an agent with that id does not exist
            or is not associated with the user.
        """
        request_context_manager = await self.async_v1_delete(f"/agents/{agent_id!s}")
        async with request_context_manager as response:
            parsed = await self.async_parsed_response(response, AgentResponse)
        return parsed.data

    def update_agent(self, agent_id: str | UUID, name: str, model: str, instructions: str) -> Agent:
        """Update all agent attributes.

        Args:
            agent_id: The id of the agent to update.
            name: The new/existing name of the agent.
            model: The new/existing model of the agent.
            instructions: The new/existing instructions of the agent.

        Returns:
            The updated agent.

        Raises:
            Exception: If an agent with that id does not exist
            or is not associated with the user.
        """
        json_data = {"name": name, "model": model, "instructions": instructions}
        response = self.v1_put(f"/agents/{agent_id!s}", json_data=json_data)
        parsed = self.parsed_response(response, AgentResponse)
        return parsed.data

    async def async_update_agent(self, agent_id: str | UUID, name: str, model: str, instructions: str) -> Agent:
        """Update all agent attributes.

        Args:
            agent_id: The id of the agent to update.
            name: The new/existing name of the agent.
            model: The new/existing model of the agent.
            instructions: The new/existing instructions of the agent.

        Returns:
            The updated agent.

        Raises:
            Exception: If an agent with that id does not exist
            or is not associated with the user.
        """
        json_data = {"name": name, "model": model, "instructions": instructions}
        request_context_manager = await self.async_v1_put(f"/agents/{agent_id!s}", json_data=json_data)
        async with request_context_manager as response:
            parsed = await self.async_parsed_response(response, AgentResponse)
        return parsed.data

    def get_chunks(self, document_id: str | UUID) -> list[Chunk]:
        """Get chunks for a document."""
        response = self.v2_get(f"/chunks?document_id={document_id!s}")
        parsed = self.parsed_response(response, ChunkResponseList)
        return parsed.data

    def post_chunks_db(self, batch: list[ChunkCreate]) -> list[str]:
        """Create chunks for a document."""
        # The Brain API only allows batches of 50 chunks at a time
        response = self.v2_post(
            "/chunks",
            json_data=[item.model_dump() for item in batch],  # pyright: ignore[reportArgumentType]
            timeout=900,
        )
        parsed = self.parsed_response(response, CreateChunksResponse)
        return parsed.data

    async def async_start_batch(
        self,
        batch_requests: list[BatchRequest],
        tags: Optional[dict[str, str]] = None,
        fast: bool = False,
    ) -> BatchStatusResponse:
        """Start a batch."""
        cleaned_requests: list[BatchRequest] = []
        for item in batch_requests:
            if inspect.isclass(item.response_format) and issubclass(item.response_format, BaseModel):
                item.response_format = item.response_format.model_json_schema()
            cleaned_requests.append(item)
        json_data: dict[str, Any] = {
            "batch_requests": [item.model_dump() for item in cleaned_requests],
            "tags": tags,
            "fast": fast,
        }
        request_context_manager = await self.async_v1_post(
            "/batches",
            json_data=json_data,
            timeout=900,
        )
        async with request_context_manager as response:
            parsed = await self.async_parsed_response(response, BatchRetrieveStatusResponse)
        return parsed.data

    # Batches
    def start_batch(
        self,
        batch_requests: list[BatchRequest],
        tags: Optional[dict[str, str]] = None,
        fast: bool = False,
    ) -> BatchStatusResponse:
        """Start a batch."""
        cleaned_requests: list[BatchRequest] = []
        for item in batch_requests:
            if inspect.isclass(item.response_format) and issubclass(item.response_format, BaseModel):
                item.response_format = item.response_format.model_json_schema()
            cleaned_requests.append(item)
        json_data: dict[str, Any] = {
            "batch_requests": [item.model_dump() for item in cleaned_requests],
            "tags": tags,
            "fast": fast,
        }
        response = self.v1_post(
            "/batches",
            json_data=json_data,
            timeout=900,
        )
        parsed = self.parsed_response(response, BatchRetrieveStatusResponse)
        return parsed.data

    def cancel_batch(self, batch_id: UUID | str) -> BatchStatusResponse:
        """Cancel a batch."""
        response = self.v1_post(f"/batches/{batch_id}/cancel")
        parsed = self.parsed_response(response, BatchRetrieveStatusResponse)
        return parsed.data

    def batch_status_with_response(self, batch_id: UUID | str) -> BatchStatusResponse:
        """Get the status and response (if finished) of a batch."""
        response = self.v1_get(f"/batches/{batch_id}")
        parsed = self.parsed_response(response, BatchRetrieveStatusResponse)
        return parsed.data

    async def async_batch_status_with_response(self, batch_id: UUID | str) -> BatchStatusResponse:
        """Get the status and response (if finished) of a batch asynchronously."""
        response = await self.async_v1_get(f"/batches/{batch_id}")
        parsed = await self.async_parsed_response(response, BatchRetrieveStatusResponse)
        return parsed.data

    def get_batches(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[list[BatchStatusEnum]] = None,
        tags: Optional[Json[Any]] = None,
    ) -> list[BatchMetadataResponse]:
        """Get all batches for the user with given status and within given date range."""
        response = self.v1_get(
            "/batches",
            params={"start_date": start_date, "end_date": end_date, "status": status, "tags": tags},
            timeout=90,  # TODO: figure out why this was timing out at 30 seconds for Aashika
        )
        parsed = self.parsed_response(response, BatchMetadataResponseList)
        return parsed.data

    def fast_batch(
        self,
        batch_requests: list[BatchRequest],
        tags: Optional[dict[str, str]] = None,
        timeout: int = 900,
    ) -> list[FastBatchResponse]:
        """Submit a batch and get it back fast. This is 2x costlier and impacts our rate limits."""

        cleaned_requests: list[BatchRequest] = []
        for item in batch_requests:
            if inspect.isclass(item.response_format) and issubclass(item.response_format, BaseModel):
                item.response_format = item.response_format.model_json_schema()
            cleaned_requests.append(item)
        json_data: dict[str, Any] = {"batch_requests": [item.model_dump() for item in cleaned_requests], "tags": tags}
        response = self.v1_post(
            "/batches/fast",
            json_data=json_data,
            timeout=timeout,
        )
        parsed = self.parsed_response(response, FastBatchResponseList)
        return parsed.data

    def fast_batch_stream(
        self,
        batch_requests: list[BatchRequest],
        tags: Optional[dict[str, str]] = None,
        timeout: int = 900,
    ) -> sseclient.SSEClient:
        """Submit a batch and get it back fast, streaming the response. This is 2x costlier and impacts our rate limits."""
        cleaned_requests: list[BatchRequest] = []
        for item in batch_requests:
            if inspect.isclass(item.response_format) and issubclass(item.response_format, BaseModel):
                item.response_format = item.response_format.model_json_schema()
            cleaned_requests.append(item)
        json_data: dict[str, Any] = {
            "batch_requests": [item.model_dump() for item in cleaned_requests],
            "tags": tags,
            "stream": True,
        }
        response = self.v2_post(
            "/batches/fast",
            json_data=json_data,
            stream=True,
            timeout=timeout,
        )
        response = cast(Generator[Any, Any, Any], response)  # makes type checker happy
        return sseclient.SSEClient(response)

    # Auth

    def get_new_user_token(self, overwrite: Optional[bool] = False) -> IssueTokenInfo:
        """Get a new user token."""
        response = self.v1_post(f"/auth/user_token?overwrite={overwrite!s}")
        parsed = self.parsed_response(response, IssueTokenResponse)
        return parsed.data

    # Tokens
    def list_tokens(self) -> list[GetTokenResponseInfo]:
        """List all tokens."""
        response = self.v1_get("/tokens")
        parsed = self.parsed_response(response, GetTokenResponseList)
        return parsed.data

    def get_token(self, token_id: str) -> GetTokenResponseInfo:
        """Get a token by id."""
        response = self.v1_get(f"/tokens/{token_id}")
        parsed = self.parsed_response(response, GetTokenResponse)
        return parsed.data

    def create_token(self, name: str, token_type: str, user_email: str, rate_limit: str) -> IssueTokenInfo:
        """Create a token."""
        json_data = {"name": name, "token_type": token_type, "user_email": user_email, "rate_limit": rate_limit}
        response = self.v1_post("/tokens", json_data=json_data)
        parsed = self.parsed_response(response, IssueTokenResponse)
        return parsed.data

    def update_token(self, token_id: str, rate_limit: str) -> str:
        """Update a token's rate limit."""
        json_data = {"token_id": token_id, "rate_limit": rate_limit}
        response = self.v1_patch(f"/tokens/{token_id}", json_data=json_data)
        parsed = self.parsed_response(response, UpdateTokenResponse)
        return parsed.data

    def delete_token(self, token_id: str) -> str:
        """Delete a token."""
        response = self.v1_delete(f"/tokens/{token_id}")
        parsed = self.parsed_response(response, DeleteTokenResponse)
        return parsed.data

    def get_token_usage(self, token_id: str) -> TokenUsage:
        """Get token usage."""
        response = self.v1_get(f"/tokens/usage/{token_id}")
        parsed = self.parsed_response(response, TokenUsageResponse)
        return parsed.data

    def get_all_token_usage(self) -> list[TokenUsage]:
        """Get all token usage."""
        response = self.v1_get("/tokens/usage")
        parsed = self.parsed_response(response, TokenUsageResponseList)
        return parsed.data
