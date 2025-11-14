import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import time

DF_NEWS_LINK = "../data/BID.csv"

def init_driver():
    options = webdriver.EdgeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Edge(options=options)
    return driver

def crawl_content(url):
    print("Crawling:", url)
    driver = init_driver()
    driver.get(url)
    time.sleep(2)

    try:
        # ⚠ Thay class bằng class thật của trang
        contents = driver.find_element(
            By.XPATH, "//*[@class='detail-content afcbc-body']"
        )
        paragraphs = contents.find_elements(By.TAG_NAME, "p")

        print("Found:", len(paragraphs), "paragraphs")

        text = "\n".join([p.text.strip() for p in paragraphs if p.text.strip() != ""])

    except NoSuchElementException:
        text = ""

    driver.quit()
    return text


# TEST
df = pd.read_csv(DF_NEWS_LINK)
df['content'] = ""
for i, url in enumerate(df['URL']):
    content = crawl_content(url)
    # add to df 
    if content:
        df['content'].at[i] = content

# Save df wwith content 
df.to_csv("../data/BID_with_content.csv", index=False)
