from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

TEAMS = {
    "Atlanta FaZe",
    "Boston Breach",
    "Carolina Royal Ravens",
    "Las Vegas Legion",
    "Los Angeles Guerrillas",
    "Los Angeles Thieves",
    "Miami Heretics",
    "Minnesota RØKKR",
    "New York Subliners",
    "OpTic Texas",
    "Seattle Surge",
    "Toronto Ultra",
}

# Setup WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Open a webpage
driver.get("https://www.breakingpoint.gg/cdl/teams-and-players")

# Example: Click a button by its ID

button_xpath = "/html/body/div[1]/div/div[1]/main/div/div[1]/div/div/div[2]/div[3]/div[1]/button[2]/div"
button = WebDriverWait(driver, 20).until(
    EC.element_to_be_clickable((By.XPATH, button_xpath))
)
button.click()


# Wait for the page to load or for specific elements to become available
# You can use time.sleep() or the more sophisticated WebDriverWait

# Find all player elements (adjust the selector as needed)
players_container_xpath = "/html/body/div[1]/div/div[1]/main/div/div[1]/div/div/div[2]/div[3]/div[3]/div/div[2]"
players_container = WebDriverWait(driver, 20).until(
    EC.visibility_of_element_located((By.XPATH, players_container_xpath))
)

# Collect all 'a' tags within the specified container
player_links = players_container.find_elements(By.TAG_NAME, "a")

# Store hrefs because clicking will change the current DOM
player_links_dict = {}

for link in player_links:
    abs_link = link.get_attribute("href")
    player_links_dict[abs_link.split("/")[-1]] = abs_link

# print(player_links_dict)

player_links_dict = {"2ReaL": "https://www.breakingpoint.gg/players/2ReaL"}

allPlayerStats = []

for player, player_link in player_links_dict.items():

    driver.get(player_link)

    cookies_button_xpath = '//*[@id="__next"]/div/div[1]/main/div[2]/div/div[3]/button'
    cookies_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, cookies_button_xpath))
    )
    cookies_button.click()

    mathes_button_xpath = "mantine-r3-tab-matches"
    matches_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.ID, mathes_button_xpath))
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", matches_button)
    matches_button.click()

    completed_mathes_button_xpath = (
        '//*[@id="mantine-r3-panel-matches"]/div[1]/button[2]'
    )
    completed_mathes_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, completed_mathes_button_xpath))
    )
    completed_mathes_button.click()

    time.sleep(10)

    matches_sections_container = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located(
            (By.XPATH, '//*[@id="mantine-r3-panel-matches"]/div[2]')
        )
    )

    matches_sections = matches_sections_container.find_elements(By.XPATH, "./div")

    match_links = []

    for match_section in matches_sections:
        innerDivs = match_section.find_elements(By.XPATH, "./div")
        matches = innerDivs[0].find_elements(By.XPATH, "./div")[1:]
        for match in matches:
            match_link = match.find_element(By.XPATH, "./a").get_attribute("href")
            match_links.append(match_link)

    for match_link in match_links:
        driver.get(match_link)
        time.sleep(1)
        match_id = match_link.split("/")[4]
        maps_section = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    '//*[@id="__next"]/div/div[1]/main/div/div[1]/div/div/div[2]/div[3]/div[2]',
                )
            )
        )

        date = driver.find_element(
            By.XPATH,
            '//*[@id="__next"]/div/div[1]/main/div/div[1]/div/div/div[2]/div[1]/div/div[1]/div[2]/div/div',
        )
        date = date.text.split(",")[0]

        maps = maps_section.find_elements(By.XPATH, "./div")

        for i, map in enumerate(maps):
            map_id = map.get_attribute("id")
            if map.get_attribute("id") == "mantine-r3-panel-overview":  # overall
                table = map.find_element(By.XPATH, "./table/tbody")
                rows = table.find_elements(By.XPATH, "./tr")
                currentTeam = ""
                for i, row in enumerate(rows):
                    try:
                        tds = row.find_elements(By.XPATH, "./td")
                        nameCol = tds[0].find_element(By.XPATH, "./a/div")
                        innerTxt = nameCol.text
                        if innerTxt in TEAMS:
                            currentTeam = innerTxt
                            continue
                        print(currentTeam)
                    except:
                        continue

                    if innerTxt == player:
                        kills = row.find_element(By.XPATH, "./td[2]").text
                        deaths = row.find_element(By.XPATH, "./td[3]").text
                        kd = row.find_element(By.XPATH, "./td[4]").text
                        dmg = row.find_element(By.XPATH, "./td[6]").text
                        allPlayerStats.append(
                            {
                                "Match_ID": match_id,
                                "Player": player,
                                "Mode": "overall",
                                "Date": date,
                                "Kills": kills,
                                "Deaths": deaths,
                                "KD": kd,
                                "Damage": dmg,
                                "Team": currentTeam,
                            }
                        )

print(allPlayerStats)

# print(f"{player} : {match_links}")


driver.quit()
