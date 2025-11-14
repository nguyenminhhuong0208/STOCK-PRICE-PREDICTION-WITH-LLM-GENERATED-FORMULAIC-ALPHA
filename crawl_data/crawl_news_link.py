import pandas as pd 
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import time
from datetime import datetime
# Khởi tạo Edge driver
def init_driver():
    options = webdriver.EdgeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Edge(options=options)
    return driver

def crawl_data(url, num_pages):
    driver = init_driver()
    driver.get(url)
    crawling = True 
    data = []
    print(f"Start scrawling {num_pages} pages from: {url}")

    for i in range(num_pages):
        print(f"Crawling page {i+1}...")
        try:
            posts = driver.find_elements(By.XPATH, '//*[@id="divTopEvents"]/ul/li')
            # try another xpath if not found any posts 
            if len(posts) == 0: 
                posts = driver.find_elements(By.XPATH, '//*[@id="divEvents"]/ul/li')
            print(f"Found {len(posts)} posts on page {i+1}")
            for post in posts:
                try:
                    time_text = post.find_element(By.XPATH, './span').text.strip()
                    # Check time stop condition 
                    cutoff_date = datetime(2024, 1, 1)
                    post_date_str = time_text.split(' ')[0][1:]
                    print(f"Post date string: {post_date_str}")
                    # Chuyển text sang đối tượng datetime
                    post_date = datetime.strptime(post_date_str, '%d/%m/%Y')
                    if post_date < cutoff_date:
                        crawling = False
                        break
                    a_tag = post.find_element(By.XPATH, './a')
                    title_text = a_tag.text.strip()
                    link_url = a_tag.get_attribute("href")  # lấy URL

                    data.append({
                        'Thời gian': time_text,
                        'Tiêu đề': title_text,
                        'URL': link_url
                    })
                except NoSuchElementException:
                    continue

            if not crawling:
                print("Reached cutoff date. Stopping crawl.")
                break
            # Nhấn nút "Sau" (AJAX load, không đổi URL)
            next_button = driver.find_element(By.XPATH, '//*[@id="aNext"]')
            if next_button.is_displayed() and next_button.is_enabled():
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(2)  # chờ dữ liệu mới load
            else:
                print("Không còn nút 'Sau'. Kết thúc.")
                break
        except NoSuchElementException:
            print("Không tìm thấy bài viết. Dừng lại.")
            break

    driver.quit()
    return pd.DataFrame(data)


URL = "https://cafef.vn/du-lieu/hose/ctg-ngan-hang-thuong-mai-co-phan-cong-thuong-viet-nam.chn"
df = crawl_data(URL, 200)
# Save csv file 
df.to_csv("../data/CTG.csv")
print("Done!")