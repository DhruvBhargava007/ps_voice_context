"""Utility functions for the Brain Platform Client."""

from typing import Any


# right now gemini only supports the following types of json schemas.
def flatten_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Flatten a JSON schema by inlining $ref references."""
    result: dict[str, Any] = schema.copy()

    # If there are no definitions, return the schema as is
    if "$defs" not in result:
        return result

    definitions = result.pop("$defs", {})

    # Function to recursively resolve references
    def resolve_refs(obj: dict[str, Any]) -> dict[str, Any]:
        if isinstance(obj, dict):
            if "$ref" in obj and obj["$ref"].startswith("#/$defs/"):
                # Extract the definition name from the reference
                def_name = obj["$ref"].split("/")[-1]
                if def_name in definitions:
                    # Replace the reference with the actual definition
                    # but preserve any other keys in the original object
                    resolved = definitions[def_name].copy()
                    # Remove the $ref key
                    obj_without_ref = {k: v for k, v in obj.items() if k != "$ref"}
                    # Merge the resolved definition with any other keys
                    resolved.update(obj_without_ref)
                    return resolved
            # Process all dictionary items
            return {k: resolve_refs(v) if isinstance(v, dict) else v for k, v in obj.items()}

    # Resolve all references in the schema
    return resolve_refs(result)
