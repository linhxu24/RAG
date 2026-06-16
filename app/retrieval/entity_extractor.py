import re

from rapidfuzz import fuzz, process


class EntityExtractor:
    def extract(
        self,
        query: str,
        product_names: list[str],
        service_names: list[str],
        *,
        multiple: bool = False,
    ) -> dict[str, list[str]]:
        return {
            "products": self._match(query, product_names, multiple=multiple),
            "services": self._match(query, service_names, multiple=multiple),
        }

    def _match(self, query: str, candidates: list[str], *, multiple: bool) -> list[str]:
        if not candidates:
            return []
        normalized_query = query.lower()
        direct = [candidate for candidate in candidates if candidate.lower() in normalized_query]
        if direct:
            return self._dedupe(direct if multiple else direct[:1])

        segments = (
            re.split(r"\s+(?:và|voi|với|vs\.?|so với|hay)\s+", query, flags=re.IGNORECASE)
            if multiple
            else [query]
        )
        matches: list[str] = []
        for segment in segments:
            match = process.extractOne(segment, candidates, scorer=fuzz.WRatio)
            if match and match[1] >= 62:
                matches.append(str(match[0]))
        if not matches:
            broad = process.extract(query, candidates, scorer=fuzz.partial_ratio, limit=3)
            matches = [str(item[0]) for item in broad if item[1] >= 75]
        return self._dedupe(matches if multiple else matches[:1])

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))
