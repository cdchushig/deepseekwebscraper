import asyncio
import json
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy

class NewsArticle(BaseModel):
    title: str
    date: str
    summary: str
    content: str  # Contenido completo
    url: str  # URL del artículo

async def main():
    llm_strategy_links = LLMExtractionStrategy(
        provider="ollama/deepseek-llm:7b",
        api_token="none",
        schema=json.dumps({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string", "format": "uri"}  # 📌 Validación de URL
                }
            }
        }),
        extraction_type="schema",
        instruction="Extrae los títulos y URLs de los artículos en la portada. Asegúrate de que las URLs sean absolutas y comiencen con 'http'.",
        input_format="markdown",
    )

    crawl_config_links = CrawlerRunConfig(
        extraction_strategy=llm_strategy_links,
        cache_mode=CacheMode.BYPASS
    )

    browser_cfg = BrowserConfig(headless=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result_links = await crawler.arun(url="https://www.marca.com", config=crawl_config_links)

        if not result_links.success:
            print("Error al extraer enlaces:", result_links.error_message)
            return

        news_links = json.loads(result_links.extracted_content)
        print("🔗 Enlaces extraídos:", news_links)

        news_data = []
        for news in news_links:
            news_url = news.get("url")
            news_title = news.get("title", "Sin título")

            if not news_url or not news_url.startswith("http"):  # 🛑 Verificar URL válida
                print(f"❌ URL inválida para {news_title}, saltando...")
                continue

            print(f"🔍 Extrayendo contenido de: {news_title} ({news_url})")

            llm_strategy_content = LLMExtractionStrategy(
                provider="ollama/deepseek-llm:7b",
                api_token="none",
                schema=NewsArticle.schema_json(),
                extraction_type="schema",
                instruction="Extrae el título, fecha, resumen y contenido completo del artículo.",
                input_format="markdown",
            )

            crawl_config_content = CrawlerRunConfig(
                extraction_strategy=llm_strategy_content,
                cache_mode=CacheMode.BYPASS
            )

            result_content = await crawler.arun(url=news_url, config=crawl_config_content)

            if result_content.success:
                article_data = json.loads(result_content.extracted_content)
                article_data["url"] = news_url
                news_data.append(article_data)
            else:
                print(f"❌ Error al extraer contenido de {news_url}: {result_content.error_message}")

        print("\n✅ Noticias extraídas:")
        for article in news_data:
            print(f"- {article['title']} ({article['date']})")
            print(f"  Resumen: {article['summary']}")
            print(f"  Contenido:\n{article['content'][:500]}...")
            print(f"  URL: {article['url']}\n")

if __name__ == "__main__":
    asyncio.run(main())
