from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from datetime import datetime

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


# Function to scroll to the element with an offset
def scroll_to_element_with_offset(driver, element, offset):
    y = element.location["y"] - offset
    driver.execute_script(f"window.scrollTo(0, {y});")


# Setup WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Open a webpage
driver.get("https://www.breakingpoint.gg/cdl/teams-and-players")

# Example: Click a button by its ID

button_xpath = "/html/body/div[1]/div/div[1]/main/div/div[1]/div/div/div[2]/div[3]/div[1]/button[2]/div"
button = WebDriverWait(driver, 3).until(
    EC.element_to_be_clickable((By.XPATH, button_xpath))
)
button.click()


# Wait for the page to load or for specific elements to become available
# You can use time.sleep() or the more sophisticated WebDriverWait

# Find all player elements (adjust the selector as needed)
players_container_xpath = "/html/body/div[1]/div/div[1]/main/div/div[1]/div/div/div[2]/div[3]/div[3]/div/div[2]"
players_container = WebDriverWait(driver, 3).until(
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

player_links_dict = {"aBeZy": "https://www.breakingpoint.gg/players/aBeZy"}

allPlayerStats = []

for player, player_link in player_links_dict.items():

    driver.get(player_link)

    cookies_button_xpath = "//button[contains(., 'Accept')]"
    cookies_button = WebDriverWait(driver, 3).until(
        EC.element_to_be_clickable((By.XPATH, cookies_button_xpath))
    )
    cookies_button.click()

    mathes_button_xpath = "(//button[contains(., 'Matches')])[3]"
    matches_button = WebDriverWait(driver, 3).until(
        EC.element_to_be_clickable((By.XPATH, mathes_button_xpath))
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", matches_button)
    matches_button.click()

    completed_mathes_button_xpath = "//button[contains(., 'Completed Matches')]"
    completed_mathes_button = WebDriverWait(driver, 3).until(
        EC.element_to_be_clickable((By.XPATH, completed_mathes_button_xpath))
    )
    completed_mathes_button.click()

    time.sleep(1.5)

    matches_sections_container = WebDriverWait(driver, 3).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "div.mantine-Stack-root.mantine-4bos9j")
        )
    )

    matches_sections = matches_sections_container.find_elements(
        By.XPATH, "./div[contains(., '2024')]"
    )

    # print(len(matches_sections))

    match_links = []

    for match_section in matches_sections:
        innerDiv = match_section.find_element(By.XPATH, "./div")
        matchesDiv = innerDiv.find_elements(By.XPATH, "./div")[1]
        matches = matchesDiv.find_elements(By.XPATH, "./div")[1:]
        for match in matches:
            match_link = match.find_element(By.XPATH, "./a").get_attribute("href")
            match_links.append(match_link)

    # print(match_links)
    for matchChecked, match_link in enumerate(match_links):
        if matchChecked == 3:
            break
        try:
            driver.get(match_link)
            time.sleep(1.5)
            match_id = match_link.split("/")[4]
            print("Match id: ", match_id)

            date = WebDriverWait(driver, 3).until(
                EC.visibility_of_element_located(
                    (
                        By.XPATH,
                        "//div[contains(text(), '2024') and (contains(text(), 'pm') or contains(text(), 'am'))]",
                    )
                )
            )

            date = date.text

            # print("Date RAW: ", date)

            # Convert the scraped date string to a datetime object
            date_obj = datetime.strptime(date, "%Y %m/%d %I:%M %p")

            # Format the datetime object into a standardized string
            formatted_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")

            # print("Date: ", formatted_date)

            buttons_with_map = WebDriverWait(driver, 3).until(
                EC.presence_of_all_elements_located(
                    (By.XPATH, "//button[div[contains(text(), 'Map')]]")
                )
            )

            overview_button = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        '//button[contains(., "Overview")]',
                    )
                )
            )

            buttons_with_map.insert(0, overview_button)
            scroll_to_element_with_offset(
                driver, overview_button, 200
            )  # Adjust the offset as needed
            time.sleep(0.2)

            for mapNum, button in enumerate(buttons_with_map):
                button.click()
                time.sleep(0.2)
                # print("Button clicked")
                teams = []
                playersInMatch = []
                table_body = WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located(
                        (
                            By.XPATH,
                            "//table/tbody",
                        )
                    )
                )

                rows = table_body.find_elements(By.XPATH, "./tr")

                for i, row in enumerate(rows):
                    tds = row.find_elements(By.XPATH, "./td")
                    nameCol = tds[0].find_element(By.XPATH, "./a/div")
                    innerTxt = nameCol.get_attribute("innerText")
                    if i == 0 or i == 5:
                        teams.append(innerTxt)
                    else:
                        if mapNum == 0:  # overall
                            mode = "overall"
                            kills = row.find_element(By.XPATH, "./td[2]").get_attribute(
                                "innerText"
                            )
                            deaths = row.find_element(
                                By.XPATH, "./td[3]"
                            ).get_attribute("innerText")
                            kd = row.find_element(By.XPATH, "./td[4]").get_attribute(
                                "innerText"
                            )
                            dmg = row.find_element(By.XPATH, "./td[6]").get_attribute(
                                "innerText"
                            )
                            hillTime = None
                            firstBloods = None
                            ticks = None

                        elif mapNum == 1 or mapNum == 4:  # hardpoint
                            mode = "HardPoint"
                            kills = row.find_element(By.XPATH, "./td[2]").get_attribute(
                                "innerText"
                            )
                            deaths = row.find_element(
                                By.XPATH, "./td[3]"
                            ).get_attribute("innerText")
                            kd = row.find_element(By.XPATH, "./td[4]").get_attribute(
                                "innerText"
                            )
                            dmg = row.find_element(By.XPATH, "./td[6]").get_attribute(
                                "innerText"
                            )
                            hillTime = row.find_element(
                                By.XPATH, "./td[7]"
                            ).get_attribute(
                                "innerText"
                            )  # seconds
                            firstBloods = None
                            ticks = None
                        else:
                            mode = None
                            kills = None
                            deaths = None
                            kd = None
                            dmg = None
                            hillTime = None
                            firstBloods = None
                            ticks = None

                        curPlayerStats = {
                            "Match_ID": match_id,
                            "Player": player,
                            "Mode": mode,
                            "Date": date,
                            "Kills": kills,
                            "Deaths": deaths,
                            "KD": kd,
                            "Damage": dmg,
                            "HillTime": hillTime,
                            "FirstBloods": firstBloods,
                            "Ticks": ticks,
                        }
                        playersInMatch.append(innerTxt)
                # print(playersInMatch)
                isInFirstTeam = False
                for i in range(len(playersInMatch)):
                    if playersInMatch[i] == player and i <= 3:
                        isInFirstTeam = True

                playersInMatch.remove(player)

                if isInFirstTeam:
                    curPlayerStats["PlayerTeam"] = teams[0]
                    curPlayerStats["TeamMate1"] = playersInMatch[0]
                    curPlayerStats["TeamMate2"] = playersInMatch[1]
                    curPlayerStats["TeamMate3"] = playersInMatch[2]
                    curPlayerStats["EnemyTeam"] = teams[1]
                    curPlayerStats["Enemy1"] = playersInMatch[3]
                    curPlayerStats["Enemy2"] = playersInMatch[4]
                    curPlayerStats["Enemy3"] = playersInMatch[5]
                    curPlayerStats["Enemy4"] = playersInMatch[6]
                else:
                    curPlayerStats["PlayerTeam"] = teams[1]
                    curPlayerStats["TeamMate1"] = playersInMatch[4]
                    curPlayerStats["TeamMate2"] = playersInMatch[5]
                    curPlayerStats["TeamMate3"] = playersInMatch[6]
                    curPlayerStats["EnemyTeam"] = teams[0]
                    curPlayerStats["Enemy1"] = playersInMatch[0]
                    curPlayerStats["Enemy2"] = playersInMatch[1]
                    curPlayerStats["Enemy3"] = playersInMatch[2]
                    curPlayerStats["Enemy4"] = playersInMatch[3]

                allPlayerStats.append(curPlayerStats)
                # print(teams)

        except Exception as e:
            print(e)
            continue

print(allPlayerStats)

# print(f"{player} : {match_links}")


driver.quit()
