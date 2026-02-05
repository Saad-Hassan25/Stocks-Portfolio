import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        return driver
    except Exception as e:
        logging.error(f"Failed to initialize Chrome driver: {e}")
        return None

def scrape_index(index_name):
    """
    Scrapes stock data for a given index (e.g., KMI30, KSE100) from Sarmaaya.pk
    """
    url = f"https://sarmaaya.pk/indexes/{index_name}"
    logging.info(f"Starting scrape for {index_name} at {url}")
    
    driver = get_driver()
    if not driver:
        return None

    try:
        driver.get(url)
        
        # Wait for table to load
        try:
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "tbody tr"))
            )
        except Exception as e:
            logging.error(f"Timeout waiting for data on {url}: {e}")
            return None

        # Get HTML and parse
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.select("tbody tr")
        
        data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue
            
            try:
                # Extract text and clean data
                stock = cols[0].get_text(strip=True)
                
                # Handling potential formatting issues (commas, %, etc)
                weight_text = cols[2].get_text(strip=True).replace('%', '').replace(',', '')
                price_text = cols[3].get_text(strip=True).replace(',', '')
                
                weight = float(weight_text) if weight_text else 0.0
                price = float(price_text) if price_text else 0.0
                
                data.append({
                    "Stock": stock,
                    f"Weight_{index_name}": weight,
                    f"Price_{index_name}": price
                })
            except ValueError as ve:
                logging.warning(f"Skipping row due to parse error: {ve}")
                continue

        df = pd.DataFrame(data)
        logging.info(f"Successfully scraped {len(df)} rows for {index_name}")
        return df

    except Exception as e:
        logging.error(f"Error scraping {index_name}: {e}")
        return None
    finally:
        driver.quit()

def fetch_all_data():
    """
    Fetches both KSE100 and KMI30 data.
    """
    kse100 = scrape_index("KSE100")
    kmi30 = scrape_index("KMI30")
    
    return kse100, kmi30
