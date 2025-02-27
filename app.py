import os
import asyncio
import json
from pydantic import BaseModel, Field
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy

class NewsArticle(BaseModel):
    title: str
    date: str
    summary: str  # Nuevo campo para el resumen

async def main():
    llm_strategy = LLMExtractionStrategy(
        provider="ollama/deepseek-llm:7b",
        api_token="none",
        schema=NewsArticle.schema_json(),
        extraction_type="schema",
        instruction=(
            "Extrae todos los artículos de noticias de la portada, obteniendo el título, "
            "la fecha y un resumen corto del contenido principal."
        ),
        chunk_token_threshold=1500,  # Aumentado para más contexto
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": 0.0, "max_tokens": 500}  # Ajustado para resúmenes cortos
    )

    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS
    )

    browser_cfg = BrowserConfig(headless=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(
            url="https://www.marca.com",
            config=crawl_config
        )

        if result.success:
            data = json.loads(result.extracted_content)
            print("Noticias extraídas:")
            for article in data:
                print(f"- {article['title']} ({article['date']})")
                print(f"  Resumen: {article['summary']}\n")

            llm_strategy.show_usage()
        else:
            print("Error:", result.error_message)

if __name__ == "__main__":
    asyncio.run(main())
