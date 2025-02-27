import asyncio
import json
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy

# Definir la estructura de datos esperada
class NewsArticle(BaseModel):
    title: str
    date: str
    summary: str
    content: str  # Nuevo campo para el contenido completo
    url: str  # Para guardar el enlace de la noticia

async def main():
    # 🔹 1️⃣ Extraer enlaces de las noticias desde la portada
    llm_strategy_links = LLMExtractionStrategy(
        provider="ollama/deepseek-llm:7b",
        api_token="none",
        schema=json.dumps({
            "type": "array",
            "items": {"type": "object", "properties": {"title": {"type": "string"}, "url": {"type": "string"}}}
        }),
        extraction_type="schema",
        instruction="Extrae los títulos de los artículos y sus URLs en la portada.",
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
        print("Noticias encontradas:", news_links)

        # 🔹 2️⃣ Visitar cada enlace y extraer el contenido completo
        news_data = []
        for news in news_links:
            news_url = news.get("url")
            news_title = news.get("title", "Sin título")

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
                article_data["url"] = news_url  # Guardar el enlace
                news_data.append(article_data)
            else:
                print(f"❌ Error al extraer contenido de {news_url}: {result_content.error_message}")

        # 🔹 3️⃣ Mostrar los resultados
        print("\n✅ Noticias extraídas:")
        for article in news_data:
            print(f"- {article['title']} ({article['date']})")
            print(f"  Resumen: {article['summary']}")
            print(f"  Contenido:\n{article['content'][:500]}...")  # Mostrar solo los primeros 500 caracteres
            print(f"  URL: {article['url']}\n")

if __name__ == "__main__":
    asyncio.run(main())
