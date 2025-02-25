import asyncio
from crawl4ai import *
from urllib.parse import urljoin, urlparse

def is_valid_url(url):
    """Check if the URL is valid and has http/https scheme."""
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except:
        return False

async def main():
    base_url = "https://tekrevol.com"
    async with AsyncWebCrawler() as crawler:
        # Get the initial page
        result = await crawler.arun(
            url=base_url,
            max_depth=5,  # Crawl up to 5 levels deep
            follow_links=True,  # Enable recursive crawling
            same_domain=True,  # Only follow links from the same domain
            max_pages=1000,  # Limit the total number of pages to crawl
        )
        
        # Save the crawled data to a text file
        with open('crawled_data.txt', 'w', encoding='utf-8') as file:
            # Write the main page content
            file.write(f"=== Main Page: {result.url} ===\n\n")
            file.write(result.markdown)
            
            # Crawl and save content from discovered links
            if hasattr(result, 'links') and result.links:
                file.write("\n\n=== Sub Pages ===\n\n")
                crawled_urls = set()  # Keep track of already crawled URLs
                
                for link in result.links:
                    # Convert relative URLs to absolute URLs
                    absolute_url = urljoin(base_url, link)
                    
                    # Skip if URL is invalid or already crawled
                    if not is_valid_url(absolute_url) or absolute_url in crawled_urls:
                        continue
                    
                    try:
                        sub_page = await crawler.arun(url=absolute_url)
                        if sub_page and sub_page.markdown:
                            file.write(f"\n\n--- {absolute_url} ---\n")
                            file.write(sub_page.markdown)
                            crawled_urls.add(absolute_url)
                            print(f"Successfully crawled: {absolute_url}")
                    except Exception as e:
                        print(f"Error crawling {absolute_url}: {str(e)}")
        
        print(f"Data has been saved to crawled_data.txt")
        print(f"Successfully crawled {len(crawled_urls)} pages")

if __name__ == "__main__":
    asyncio.run(main())