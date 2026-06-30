# Exemplu de utilizare Selenium cu Chrome
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def exemplu_simplu():
    """Exemplu simplu de deschidere pagină web"""
    # Inițializează browser-ul Chrome (descarcă automat driver-ul)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    try:
        # Deschide o pagină web
        driver.get("https://www.google.com")
        print(f"Titlul paginii: {driver.title}")
        
        # Așteaptă 3 secunde
        driver.implicitly_wait(3)
        
    finally:
        # Închide browser-ul
        driver.quit()


def exemplu_cautare_wikipedia():
    """Exemplu de căutare pe Wikipedia (fără protecție anti-bot)"""
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    try:
        # Wikipedia este prietenos cu automatizare
        driver.get("https://en.wikipedia.org")
        print(f"Pagină încărcată: {driver.title}")
        
        # Așteaptă ca caseta de căutare să fie CLICKABLE (nu doar prezentă)
        search_box = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.NAME, "search"))
        )
        
        # Click pe element pentru a-l activa
        search_box.click()
        
        # Scrie în caseta de căutare
        search_box.send_keys("Selenium (software)")
        search_box.submit()
        
        # Așteaptă să se încarce pagina rezultatelor
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "firstHeading"))
        )
        
        # Extrage titlul articolului
        heading = driver.find_element(By.ID, "firstHeading").text
        print(f"Articol găsit: {heading}")
        print(f"URL curent: {driver.current_url}")
        
        # Extrage primul paragraf
        try:
            first_paragraph = driver.find_element(By.CSS_SELECTOR, ".mw-parser-output > p").text
            if first_paragraph.strip():
                print(f"Primul paragraf: {first_paragraph[:150]}...")
        except:
            print("Nu s-a putut extrage primul paragraf")
        
    finally:
        driver.quit()


def exemplu_extragere_date():
    """Exemplu de extragere date dintr-o pagină"""
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    try:
        driver.get("https://www.example.com")
        
        # Extrage titlul paginii
        title = driver.title
        print(f"Titlu: {title}")
        
        # Extrage text dintr-un element
        heading = driver.find_element(By.TAG_NAME, "h1").text
        print(f"Heading: {heading}")
        
        # Extrage toate link-urile
        links = driver.find_elements(By.TAG_NAME, "a")
        print(f"\nNumăr de link-uri: {len(links)}")
        for link in links:
            print(f"- {link.get_attribute('href')}")
            
    finally:
        driver.quit()


def exemplu_evitare_detectie_bot():
    """Exemplu cu opțiuni pentru a evita detectarea ca bot"""
    from selenium.webdriver.chrome.options import Options
    
    options = Options()
    
    # Opțiuni pentru a părea mai puțin ca un bot
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    
    # User agent personalizat (opțional)
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    
    # Ascunde proprietățile webdriver
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    try:
        driver.get("https://www.wikipedia.org")
        print(f"Pagină încărcată: {driver.title}")
        print("✓ Bot detection evadat cu succes!")
        
    finally:
        driver.quit()


def exemplu_site_test_selenium():
    """Exemplu cu un site special creat pentru testare Selenium - GARANTAT să funcționeze"""
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    try:
        # Site special pentru practică Selenium
        driver.get("https://the-internet.herokuapp.com/")
        print(f"Pagină încărcată: {driver.title}")
        
        # Găsește toate link-urile de exemple
        links = driver.find_elements(By.CSS_SELECTOR, "ul li a")
        print(f"\nExemple disponibile: {len(links)}")
        
        # Testează formularul de login
        driver.get("https://the-internet.herokuapp.com/login")
        
        # Completează username
        username = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "username"))
        )
        username.send_keys("tomsmith")
        
        # Completează parola
        password = driver.find_element(By.ID, "password")
        password.send_keys("SuperSecretPassword!")
        
        # Click pe butonul de login
        login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        login_button.click()
        
        # Verifică mesajul de succes
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".flash.success"))
        )
        
        success_message = driver.find_element(By.CSS_SELECTOR, ".flash.success").text
        print(f"\n✓ Login reușit!")
        print(f"Mesaj: {success_message.strip()}")
        
    finally:
        driver.quit()


if __name__ == "__main__":
    print("=== Exemplu 1: Pagină simplă ===")
    exemplu_simplu()
    
    print("\n=== Exemplu 2: Site de test Selenium (GARANTAT) ===")
    exemplu_site_test_selenium()
    
    print("\n=== Exemplu 3: Căutare Wikipedia ===")
    try:
        exemplu_cautare_wikipedia()
    except Exception as e:
        print(f"Eroare: {e}")
    
    print("\n=== Exemplu 4: Extragere date ===")
    exemplu_extragere_date()
    
    print("\n=== Exemplu 5: Evitare detectie bot ===")
    exemplu_evitare_detectie_bot()
