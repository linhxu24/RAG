import json

from app.assets.resolver import AssetResolver
from app.generation.schemas import ChatResponse, ChatSuggestion, GeneratedResponse


class ResponseRenderer:
    def __init__(self):
        self.asset_resolver = AssetResolver()

    def resolve_assets(self, session, response: GeneratedResponse):
        asset_ids = [
            asset_id
            for item in response.result.items
            for asset_id in item.asset_ids
        ]
        doc_ids = list(
            dict.fromkeys(
                str(doc_id)
                for doc_id in [
                    *(item.doc_id for item in response.result.items),
                    *(source.doc_id for source in response.result.sources),
                ]
                if doc_id
            )
        )
        resolution = self.asset_resolver.resolve(
            session,
            self._asset_resolution_text(response),
            asset_ids=asset_ids,
            doc_ids=doc_ids,
        )
        response.result.assets = resolution.assets
        response.result.missing_assets = resolution.missing_assets
        return response

    @staticmethod
    def _asset_resolution_text(response: GeneratedResponse) -> str:
        parts = [response.result.text]
        for item in response.result.items:
            if item.data:
                parts.append(json.dumps(item.data, ensure_ascii=False, default=str))
        return "\n".join(part for part in parts if part)

    @staticmethod
    def render(
        trace_id: str,
        response: GeneratedResponse,
        *,
        suggestions: list[ChatSuggestion] | None = None,
        debug: bool = False,
        debug_data: dict | None = None,
    ) -> ChatResponse:
        return ChatResponse(
            trace_id=trace_id,
            intent=response.intent,
            answer_type=response.answer_type,
            entities=response.entities,
            answer=response.result,
            safety=response.safety,
            degraded=response.degraded,
            suggestions=suggestions or [],
            debug=dict(debug_data or {}) if debug else {},
        )
