from bs4 import BeautifulSoup
import requests

URL = 'https://www.carzone.ie/used-cars/ireland/dublin?county=Dublin'
page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')
#results = soup.findAll('a _ngcontent-cz-web-c141')
for a in soup.find_all("a", href=True):
    print("Found the URL:", a)
    
    
    

from requests_html import AsyncHTMLSession
asession = AsyncHTMLSession()

async def get_results():
    r = await asession.get(URL)
    await r.html.arender()
    return r

import asyncio
if asyncio.get_event_loop().is_running(): # Only patch if needed (i.e. running in Notebook, Spyder, etc)
    import nest_asyncio
    nest_asyncio.apply()

r = asession.run(get_results)
lang_bar = r[0].html.find('#LangBar', first=True)
print(lang_bar.html)
soup = BeautifulSoup(lang_bar.html, 'html.parser')
print(soup.prettify())



from requests_html import AsyncHTMLSession

url = 'https://www.thefreedictionary.com/love'

asession = AsyncHTMLSession()
script = """
 () => {
              if ( document.readyState === "complete" ) {
                   document.getElementsByClassName("fl_ko")[0].click();
              }
        }
         """


async def get_results():
    r = await asession.get(url)
    await r.html.arender(script=script, timeout=10, sleep=2)
    return r

r = asession.run(get_results)
for span in r[0].html.find('span.trans[style="display: inline;"]'):
    print(span, span.text)