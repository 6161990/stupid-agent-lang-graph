# pip install gdeltdoc : url 을 얻고 https://github.com/alex9smith/gdelt-doc-api
# pip install newspaper3k : url 내용이 뉴스인 경우에 크롤링(컨텐츠. 작성자 등을 분리해서 파싱할 수 있는 파이썬 라이브러리)

from gdeltdoc import GdeltDoc, Filters
from newspaper import Article

f = Filters(
    start_date = "2024-01-20",
    end_date = "2024-03-25",
    num_records = 250,
    keyword = "microsoft",
    domain = "nytimes.com",
    country = "US"
)

gd = GdeltDoc()

# Search for articles matching the filters
articles = gd.article_search(f)
url = articles.loc[1, "url"]

article = Article(url)
article.download()
article.parse()
print(article.text)


# Get a timeline of the number of articles matching the filters
# timeline = gd.timeline_search("timelinevol", f)
