"""Schemas for Brain API responses and requests."""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Type, Union
from uuid import UUID

from openai.types.audio.transcription_segment import TranscriptionSegment
from openai.types.audio.transcription_word import TranscriptionWord
from pydantic import UUID4, BaseModel, ConfigDict, Field, HttpUrl


class DocSplit(str, Enum):
    """Document chunking / splitting types."""

    CHUNK = "chunk"
    NEWLINE = "newline"


class ChunkCreate(BaseModel):
    """Schema for creating a Chunk."""

    document_id: str
    token_count: int
    chunk_index: int
    content: str
    embedding: list[float]


class Chunk(ChunkCreate):
    """Schema for a chunk."""

    id: Union[UUID, str]
    created_at: datetime


class ChunkResponseList(BaseModel):
    """Response schema for a list of Chunks."""

    data: list[Chunk]


class CreateChunksResponse(BaseModel):
    """Response schema for creating a Chunk."""

    data: list[str]


class UserConversationCostResponseData(BaseModel):
    """Cost of a conversation per user."""

    user_id: str
    cost: float


class UserConversationCostResponse(BaseModel):
    """Cost of a conversation per user."""

    data: UserConversationCostResponseData


class ConversationCostResponseData(BaseModel):
    """Cost of a conversation."""

    conversation_id: str
    tokens_used: int
    cost: float


class ConversationCostResponse(BaseModel):
    """Cost of a conversation."""

    data: ConversationCostResponseData


class Conversation(BaseModel):
    """Conversation."""

    id: str
    user_id: str
    conversation_name: Optional[str]
    tags: Union[dict[str, str], None]
    created_at: datetime
    archived: bool


class Message(BaseModel):
    """Schema for a message."""

    id: str
    conversation_id: str
    model: str
    role: str
    message: Optional[str]
    created_at: datetime
    ptc: int
    ctc: int
    context: Optional[list[str]]


class ConversationResponse(BaseModel):
    """Response from Brain API for GET /api/v1/conversations."""

    data: list[Conversation]


class ConversationWithMessagesResponseData(BaseModel):
    """Response from Brain API for GET /api/v1/conversations/{conversation_id}."""

    data: Conversation


class MessageWithImageUrls(Message):
    """Schema for messages with image URLS."""

    image_urls: Optional[list[str]]


class ConversationWithMessages(Conversation):
    """Schema for a conversation with messages."""

    messages: list[MessageWithImageUrls]


class ConversationWithMessagesResponse(BaseModel):
    """Response from Brain API for GET /api/v1/conversations/{conversation_id}."""

    data: ConversationWithMessages


# Search


class ContextInfo(BaseModel):
    """Schema for a context."""

    similarity: float
    document_id: str
    id: str


class SearchResponse(BaseModel):
    """Schema for a search response."""

    content: str
    similarity: float
    document_id: Union[UUID, str]
    chunk_index: int
    associated_content: Optional[str]


class SearchResponseList(BaseModel):
    """Response schema for a search request."""

    data: list[SearchResponse]


class SearchWithContextResponse(BaseModel):
    """Schema for a search response."""

    model_config = ConfigDict(from_attributes=True)

    chunk_content: list[SearchResponse]
    document_content: str
    document_id: Union[UUID, str]
    document_name: str
    source: str
    author: Optional[str]
    associated_content: Optional[str]
    document_created_at: datetime
    document_updated_at: datetime
    s3_key: str


class SearchWithContextResponseList(BaseModel):
    """Schema for a list of search responses with context."""

    data: list[SearchWithContextResponse]


# LLMs


class ImagesResponse(BaseModel):
    """Images Response."""

    presigned_url: str


class GenerateImageResponse(BaseModel):
    """Schema for generating images from prompts."""

    message_id: Union[UUID, str]
    conversation_id: Union[UUID, str]
    model: str
    prompt: str
    image_urls: list[ImagesResponse]


# Used for filtering; if None, returns all model types
MODEL_TYPES = Literal["text", "image", "audio", "video", "transcription"]

PROVIDER_ENUM = Literal["OpenAI", "Anthropic", "AWS", "Google", "Pattern", "XAI", "Perplexity"]


class GenerateImageResponseData(BaseModel):
    """Response from Brain API for POST /api/v1/image/generate."""

    data: GenerateImageResponse


class AiModel(BaseModel):
    """AI model."""

    class ContextWindowType(str, Enum):
        """Context window type for the model."""

        CHAR = "char"
        TOKEN = "token"

    id: str = Field(
        description="AI API model name",
    )
    display_name: str = Field(
        description="Display name for the AI model",
    )
    provider: PROVIDER_ENUM = Field(
        description="Provider of the AI model",
    )
    ai_model_type: MODEL_TYPES = Field(
        description="Type of the AI model",
    )
    context_window_type: ContextWindowType = Field(
        description="Context window type for the AI model",
        default=ContextWindowType.TOKEN.value,
    )
    context_window_size: Optional[int] = Field(
        description="LLM only. Context window size for the AI model",
    )
    supports_function_calling: Optional[bool] = Field(
        description="LLM only. Whether the AI model supports function calling",
    )
    supports_vision: Optional[bool] = Field(
        description="LLM only. Whether the AI model supports vision",
    )
    supports_streaming: Optional[bool] = Field(
        description="LLM only. Whether the AI model supports streaming",
    )
    supports_json_mode: Optional[bool] = Field(
        description="LLM only. Whether the AI model supports json mode",
    )
    supports_json_schema: Optional[bool] = Field(
        description="LLM only. Whether the AI model supports json schema",
    )
    image_resolution: Optional[str] = Field(
        description="Image only. Whether the AI model supports response format",
    )
    preferred_pointer: Optional[str] = Field(
        description="The preferred pointer for the AI model",
    )
    max_temp: Optional[float] = Field(
        description="The maximum temperature for the AI model",
    )
    active: bool = Field(
        description="Whether the AI model is active",
    )
    knowledge_cutoffs: datetime = Field(
        description="The date at which the model will no longer be updated with new knowledge",
    )
    deactivated: Optional[bool] = Field(
        description="Whether the AI model is deactivated",
    )


class AiModelWithPricing(AiModel):
    """AI model with pricing."""

    input_cost_per_million: Optional[float] = Field(
        default=None,
        description="Cost per thousand input tokens",
    )
    output_cost_per_million: Optional[float] = Field(
        default=None,
        description="Cost per thousand output tokens",
    )
    unit_type: Optional[str] = Field(
        default=None,
        description="Unit type for the pricing",
    )
    cost_per_image: Optional[float] = Field(
        default=None,
        description="Cost per image for image models",
    )


class ModelPriceCreate(BaseModel):
    """Schema for creating a new model price entry."""

    input_cost_per_million: float = Field(description="Cost per 1M input tokens")
    output_cost_per_million: float = Field(description="Cost per 1M output tokens")
    unit_type: str = Field(description="Unit type for the pricing")


class AiModelUpdate(BaseModel):
    """Schema for AI model update request."""

    context_window_type: Optional[str] = Field(default=None, description="Context window type for the AI model")
    context_window_size: Optional[int] = Field(default=None, description="Context window size for the AI model")
    active: Optional[bool] = Field(default=None, description="Whether the model is active")
    preferred_pointer: Optional[str] = Field(default=None, description="Preferred pointer for the model")
    supports_vision: Optional[bool] = Field(default=None, description="Whether the model supports vision")
    supports_streaming: Optional[bool] = Field(default=None, description="Whether the model supports streaming")
    supports_function_calling: Optional[bool] = Field(
        default=None, description="Whether the model supports function calling"
    )
    max_temp: Optional[float] = Field(default=None, description="The maximum temperature for the AI model")
    supports_json_mode: Optional[bool] = Field(default=None, description="Whether the model supports JSON mode")
    supports_json_schema: Optional[bool] = Field(default=None, description="Whether the model supports JSON schema")
    price: ModelPriceCreate = Field(description="Pricing information for the model")
    deactivated: Optional[bool] = Field(default=None, description="Whether the model is deactivated")


class AiModelWithPricingData(BaseModel):
    """Response from Brain API for GET /api/v1/models."""

    data: list[AiModelWithPricing]


class AiModelUpdateResponse(BaseModel):
    """Response from Brain API for PUT /api/v1/models/update/{model_id}."""

    data: AiModelWithPricing


class InvokeLlmResponseData(BaseModel):
    """The response from an LLM invocation."""

    llm: str = Field(
        description="The LLM model used",
    )
    response: str = Field(
        description="The response from the LLM",
    )
    conversation_id: Union[UUID, str] = Field(
        description="The conversation id",
    )
    message_id: Union[UUID, str] = Field(
        description="The message id",
    )
    collection_id: Optional[Union[UUID, str]] = Field(
        default=None,
        description="The collection id used for context",
    )
    stop_reason: Optional[str] = Field(
        default=None,
        description="The reason the LLM stopped generating text",
    )
    context: Optional[list[SearchResponse]] = Field(
        default=None,
        description="The context used for the response",
    )
    citations: Union[list[str], None] = Field(
        default=None,
        description="The citations used for the response. Only available for some models like Perplexity.",
    )
    image_url: Optional[str] = Field(default=None, description="The image URLs generated by the LLM")

    def json(self, **kwargs: Any):
        """Override json method to handle UUIDs."""
        return json.loads(json.dumps(self.model_dump(), default=str, **kwargs))


class InvokeLlmResponse(BaseModel):
    """The response from an LLM invocation."""

    data: InvokeLlmResponseData


class ToolUseResponseData(BaseModel):
    """The response from an LLM when using tool calling."""

    llm: str = Field(
        description="The LLM model used",
    )
    response: Optional[str] = Field(
        description="The response from the LLM",
    )
    conversation_id: Union[UUID, str] = Field(
        description="The conversation id",
    )
    message_id: Union[UUID, str] = Field(
        description="The message id",
    )
    collection_id: Optional[Union[UUID, str]] = Field(
        default=None,
        description="The collection id used for context",
    )
    stop_reason: Optional[str] = Field(
        default=None,
        description="The reason the LLM stopped generating text",
    )
    context: Optional[list[SearchResponse]] = Field(
        default=None,
        description="The context used for the response",
    )
    tool_use: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Result of the tool call(s)",
    )
    tools: Optional[list[str]] = Field(
        default=None,
        description="Name of the tool(s) called",
    )

    def json(self, **kwargs: Any):
        """Override json method to handle UUIDs."""
        return json.loads(json.dumps(self.model_dump(), default=str, **kwargs))


class ToolUseResponse(BaseModel):
    """The response from an LLM when using tool calling."""

    data: ToolUseResponseData


class ParameterProperties(BaseModel):
    """Describes type, description, and enum (optional) for a tool parameter."""

    type: Literal["string", "integer", "number", "boolean", "array", "object", "null"]
    description: str
    enum: Optional[list[str]] = []


class ToolParameters(BaseModel):
    """Describes the parameters for a tool."""

    type: Literal["object"] = "object"
    properties: dict[str, ParameterProperties]
    required: Optional[list[str]] = []


class Tool(BaseModel):
    """Defines what an tool should look like."""

    name: str = Field(description="The name of the tool", examples=["get_current_weather"])
    description: str = Field(
        description="The description of the tool", examples=["Get the current weather in a given location."]
    )
    parameters: ToolParameters


# Private Conversations


class PrivateConversationMessage(BaseModel):
    """Private conversation message."""

    role: str = Field(
        description="The role of the message",
    )
    content: str = Field(
        description="The message",
    )


# Collections
class CollectionUser(BaseModel):
    """Schema for a collection user."""

    id: Union[UUID, str]
    user_email: str
    created_at: Optional[datetime]


class Collection(BaseModel):
    """Schema for a collection."""

    id: str
    collection_name: str
    # choosing to use a string here instead of an enum because the Brain API
    # could add more values later
    embedding_model: str
    # if user_id is None, that means it is a public collection
    user_id: Optional[str]
    created_at: datetime
    is_owner: bool = False
    collection_users: list[CollectionUser] = []
    description: Optional[str] = None


class GetCollectionsResponse(BaseModel):
    """Response from Brain API for GET /api/v1/collections."""

    data: list[Collection]


class GetCollectionResponse(BaseModel):
    """Response from Brain API for GET /api/v1/collections/{collection_id}."""

    data: Collection


# Documents


class DocumentStatus(str, Enum):
    """Status of a document."""

    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class DocumentUpdatableFields(BaseModel):
    """Schema for a document."""

    document_name: Optional[str] = Field(
        default=None,
        description="Name of the document, primarily used for display purposes",
    )
    source: Optional[str] = Field(
        default=None,
        description="Source of the document. URL, textbook name, repository location, etc.",
    )
    author: Optional[str] = Field(
        default=None,
        description="Author of the document",
    )
    associated_content: Optional[str] = Field(
        default=None,
        description="Content associated with the document, such as SQL that answers a question",
    )


class CSVDocumentUpload(DocumentUpdatableFields):
    """Schema for a csv document upload."""

    content: str = Field(
        description="Content associated with the document, whose embedding would be generated ",
    )


class DocumentSplitFields(BaseModel):
    """Schema for defining split type / chunk size and overlap characters."""

    document_split_type: Optional[DocSplit] = Field(
        default=DocSplit.CHUNK,
        description="Chunk/split method for the uploaded document",
    )
    document_chunk_size: Optional[int] = Field(
        default=None,
        description="Chunk size for the uploaded document",
    )
    document_chunk_overlap: Optional[int] = Field(
        default=None,
        description="Chunk overlap value for the uploaded document",
    )


class Document(DocumentUpdatableFields, DocumentSplitFields):
    """Schema for a document."""

    model_config = ConfigDict(from_attributes=True)

    id: Union[UUID, str]
    document_name: str  # pyright: ignore
    collection_id: Union[UUID, str] = Field(
        description="ID of the collection the document belongs to",
    )
    created_at: datetime = Field(
        default=datetime,
        description="Upload date to youtube/creation date in wiki/published date in news",
    )
    updated_at: datetime = Field(
        default=datetime,
        description="Last updated date in wiki/news",
    )
    checksum: Union[str, None] = Field(
        default=None,
        description="MD5sum of doc, use in PUTs to see if it needs reprocessing",
    )

    s3_key: Union[str, None] = Field(
        default=None,
        description="The location of the document in S3",
    )

    status: Optional[DocumentStatus] = Field(
        default=None,
        description="Current status for the uploaded document",
    )

    created_at: datetime
    updated_at: datetime


class GetDocumentsResponse(BaseModel):
    """Response from Brain API for GET /api/v1/documents."""

    data: list[Document]


class DocumentShowResponse(BaseModel):
    """Response from Brain API for GET /api/v1/documents/{document_id}."""

    data: Document


class GetCsvResponse(BaseModel):
    """Response from Brain API for GET /api/v1/documents/csv."""

    data: list[str]


class GetDocumentResponse(BaseModel):
    """Response from Brain API for GET /api/v1/documents/{document_id}."""

    data: Document


class DocumentPresignedUrl(Document):
    """Document with presigned url."""

    presigned_url: Optional[str] = None


class DocumentPresignedUrlResponse(BaseModel):
    """Response from Brain API containing a list of documents with presigned urls."""

    data: list[DocumentPresignedUrl]


class DeleteTranscriptionResponse(BaseModel):
    """Response schema for deleting a transcription."""

    data: str


class TranscriptionResponse(BaseModel):
    """Transcription Response."""

    id: Union[UUID4, str]
    audio_file: Optional[HttpUrl] = None
    timestamps: Optional[dict[Literal["segments", "words"], Union[list[TranscriptionSegment], list[TranscriptionWord]]]] = (
        None
    )
    transcription: str
    language: Optional[str] = Field(
        description="Language code used for transcription (e.g., 'en', 'es', 'fr')",
        default=None,
    )
    user_id: Union[str, None]
    created_at: datetime
    status: str


class TranscriptionWorkerResponse(BaseModel):
    """Transcription Worker Response."""

    transcription_id: Union[UUID, str]
    status: str


class TranscriptionWorker(BaseModel):
    """Transcription Worker List."""

    data: TranscriptionWorkerResponse


class TranscriptionResponseList(BaseModel):
    """Response schema for list of audio transcription."""

    data: list[TranscriptionResponse]


class TranscriptionResponseData(BaseModel):
    """Response schema for a single audio transcription."""

    data: TranscriptionResponse


# Agents
class AgentUser(BaseModel):
    """Schema for an agent user."""

    id: Union[UUID, str]
    user_email: str
    created_at: Optional[datetime] = None


class Agent(BaseModel):
    """Schema for an agent."""

    id: Union[UUID, str]
    name: str
    user_id: Optional[str]  # as we can have public agents
    instructions: str
    model: str
    is_owner: bool = False
    agent_users: list[AgentUser] = []


class AgentResponse(BaseModel):
    """Response schema for a single agent."""

    data: Agent


class GetAgentsResponse(BaseModel):
    """Response schema for a list of agents."""

    data: list[Agent]


class EmbeddingsResponse(BaseModel):
    """Schema for embeddings response."""

    data: list[list[float]]


# Batch


class BatchStatusEnum(str, Enum):
    """Status of a batch."""

    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FINALIZING = "finalizing"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"
    CANCELING = "canceling"
    ENDED = "ended"
    FAST_PROCESSING = "fast_processing"


class BatchRequestCounts(BaseModel):
    """Batch request counts."""

    completed: int
    """Number of requests that have been completed successfully."""

    failed: int
    """Number of requests that have failed."""

    total: int
    """Total number of requests in the batch."""


class BatchStatus(BaseModel):
    """Status of the batch processing."""

    status: Literal[
        "validating",
        "failed",
        "in_progress",
        "finalizing",
        "completed",
        "expired",
        "cancelling",
        "cancelled",
        "ended",
        "canceling",
        "fast_processing",
    ] = Field(description="The status of the batch processing as reported by OpenAI")
    request_counts: BatchRequestCounts = Field(description="Request counts for the batch")


class BatchRequest(BaseModel):
    """Request for batch processing."""

    model: str = Field(description="The model to use for the message. Models can only be mixed within providers")
    message: str = Field(description="The user message used to generate a response")
    system_message: Optional[str] = Field(description="Optional system message to steer behavior", default=None)
    message_id: Optional[str] = Field(
        description="Optional message id for tracking. If not provided, one will be generated", default=None
    )
    image_url: Optional[Union[list[str], str]] = Field(description="The image URL to use for the message", default=None)
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature to use for batches. If not provided, the default is 0.1",
    )
    response_format: Optional[Union[Literal["json_object", "text"], Type[BaseModel], dict[str, Any]]] = Field(
        default="text",
        description="""
        The format to use for the response.  If set to `json_object`, the response will be constrained
        to a valid JSON object.
        """.strip(),
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="The URL to call when the batch is complete. The callback will be called with the batch_id and the status of the batch.",
    )


class BatchResponse(BaseModel):
    """Response for batch processing."""

    model: str = Field(description="The model used for the message")
    message: Union[str, list[dict[str, Any]]] = Field(description="The user message used to generate a response")
    response: str = Field(description="The response generated by the model")
    message_id: str = Field(description="Message id for tracking")
    cost: float = Field(description="The cost of the message")


class BatchRetrieveResponse(BaseModel):
    """Response from retrieve_batch_response()."""

    data: list[BatchResponse] = Field(description="The response generated by the model")


class BatchStatusResponse(BatchStatus):
    """Status of the batch processing with batch_id. For returning to user."""

    batch_id: Union[UUID, str] = Field(description="The UUID batch id used by Brain")
    batch_response: Optional[list[BatchResponse]] = Field(
        description="The response generated by the model", default=None
    )
    total_cost: Optional[float] = Field(description="The cost of the Batch", default=None)
    tags: Optional[dict[str, str]] = Field(description="Tags associated with the batch", default=None)
    callback_url: Optional[str] = Field(
        default=None,
        description="The URL to call when the batch is complete. The callback will be called with the batch_id and the status of the batch.",
    )


class BatchRetrieveStatusResponse(BaseModel):
    """Response from start_batch()."""

    data: BatchStatusResponse = Field(description="The response generated by the model")


class BatchMetadataResponse(BaseModel):
    """Response from db for batch metadata."""

    id: Union[UUID, str] = Field(description="The UUID batch id used by Brain")
    status: str = Field(description="Status of the batch")
    cost: float = Field(description="The cost of the batch")
    created_at: datetime = Field(description="Creation date of the batch")
    updated_at: datetime = Field(description="Last status update of the batch")
    request_counts: Optional[str] = Field(description="Request counts for the batch")
    input_presigned_url: str = Field(description="Link to file containing Input Prompts for requests")
    output_presigned_url: Optional[str] = Field(description="Link to file containing the Output for requests")
    tags: Optional[dict[str, str]] = Field(description="Tags associated with the batch")
    callback_url: Optional[str] = Field(
        default=None,
        description="The URL to call when the batch is complete. The callback will be called with the batch_id and the status of the batch.",
    )


class BatchMetadataResponseList(BaseModel):
    """Response from Brain API containing a list of batches."""

    data: list[BatchMetadataResponse]


class FastBatchResponse(BaseModel):
    """Fast Batch response object."""

    message_id: Optional[str] = None
    message: str
    image_urls: Optional[Union[list[str], str]] = None
    temperature: Optional[float] = None
    response: str


class FastBatchResponseList(BaseModel):
    """Fast Batch response object."""

    data: list[FastBatchResponse]


class IssueTokenInfo(BaseModel):
    """API Token Info."""

    token_name: str
    token_type: str
    user_id: str
    user_email: Union[str, None]
    token: str
    rate_limit: str


class IssueTokenResponse(BaseModel):
    """API Token Info Response."""

    data: IssueTokenInfo


class GetTokenResponseInfo(BaseModel):
    """Response to get a token."""

    token_id: str
    name: str
    token_type: str
    user_id: str
    user_email: Union[str, None]
    created_at: datetime
    rate_limit: Union[str, None]


class GetTokenResponse(BaseModel):
    """Response to get a token."""

    data: GetTokenResponseInfo


class GetTokenResponseList(BaseModel):
    """Response to get a list of tokens."""

    data: list[GetTokenResponseInfo]


class UpdateTokenResponse(BaseModel):
    """Response to update a token."""

    data: str


class DeleteTokenResponse(BaseModel):
    """Response to delete a token."""

    data: str


class WindowStats(BaseModel):
    """Stats for a window."""

    reset_time: float
    remaining: int
    limit: int


class TokenUsage(BaseModel):
    """Usage for a token."""

    token_name: str
    token_id: str
    current_usage: dict[str, WindowStats]


class TokenUsageResponse(BaseModel):
    """Response to get token usage."""

    data: TokenUsage


class TokenUsageResponseList(BaseModel):
    """Response to get all token usage."""

    data: list[TokenUsage]
