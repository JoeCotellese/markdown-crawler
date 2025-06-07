from markdown_crawler import md_crawl
#url = 'https://rickandmorty.fandom.com/wiki/Evil_Morty'
url = 'https://support.optimizely.com/hc/en-us/sections/27522568219789-Get-started-with-Personalization'
print(f'🕸️ Starting crawl of {url}')
md_crawl(
    url,
    max_depth=2,
    num_threads=5,
    base_dir='markdown',
    is_domain_match=True,
    is_base_path_match=False,
    is_debug=True,
    render_html=True  # Enable JavaScript rendering with Playwright
)