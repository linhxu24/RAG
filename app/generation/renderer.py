from app.assets.resolver import AssetResolver
from app.generation.schemas import ChatResponse, GeneratedResponse


class ResponseRenderer:
    def __init__(self):
        self.asset_resolver = AssetResolver()

    def resolve_assets(self, session, response: GeneratedResponse):
        asset_ids = [
            asset_id
            for item in response.result.items
            for asset_id in item.asset_ids
        ]
        resolution = self.asset_resolver.resolve(
            session,
            response.result.text,
            asset_ids=asset_ids,
        )
        response.result.assets = resolution.assets
        response.result.missing_assets = resolution.missing_assets
        return response

    @staticmethod
    def render(
        trace_id: str,
        response: GeneratedResponse,
        *,
        debug: bool = False,
        debug_data: dict | None = None,
    ) -> ChatResponse:
        return ChatResponse(
            trace_id=trace_id,
            intent=response.intent,
            answer_type=response.answer_type,
            answer=response.result,
            safety=response.safety,
            degraded=response.degraded,
            debug={"enabled": debug, **(debug_data or {})} if debug else {"enabled": False},
        )
