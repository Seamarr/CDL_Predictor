from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from datetime import datetime
import csv
from selenium.webdriver.chrome.options import Options
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datetime import datetime
import numpy as np

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


def initialize_driver():
    chrome_options = Options()
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--ignore-ssl-errors")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


# Function to scroll to the element with an offset
def scroll_to_element_with_offset(driver, element, offset):
    y = element.location["y"] - offset
    driver.execute_script(f"window.scrollTo(0, {y});")


# Function to check if a button contains a div with specific text
def button_contains_text(button, text):
    divs = button.find_elements(By.XPATH, ".//div")
    for div in divs:
        if text in div.text:
            return True
    return False


def calculate_kd(kills, deaths):
    if deaths == 0:
        deaths = 1  # Avoid division by zero
    kd_ratio = kills / deaths
    return round(kd_ratio, 2)


def scrape():

    # Setup WebDriver
    driver = initialize_driver()

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

    # player_links_dict = {"aBeZy": "https://www.breakingpoint.gg/players/aBeZy"}

    allPlayerStats = []

    for player, player_link in player_links_dict.items():
        # if player != "Clayster" and player != "aBeZy":
        #     continue

        driver.get(player_link)
        time.sleep(1.5)

        try:
            cookies_button_xpath = "//button[contains(., 'Accept')]"
            cookies_button = WebDriverWait(driver, 0.5).until(
                EC.element_to_be_clickable((By.XPATH, cookies_button_xpath))
            )
            cookies_button.click()
        except:
            pass

        mathes_button_xpath = (
            "(//button[contains(., 'Matches') and contains(@id, 'tab-matches')])"
        )
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

        totalMacthesToScrape = len(match_links)
        totalMatchesSuccessfullyScraped = 0

        # print(match_links)
        for matchChecked, match_link in enumerate(match_links):
            # if matchChecked == 3:
            #     break
            try:
                driver.get(match_link)
                time.sleep(1.5)
                match_id = match_link.split("/")[4]
                print(
                    f"Extracting {player}'s data for match #{match_id}... ({matchChecked+1}/{totalMacthesToScrape})"
                )

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

                successfulExtracts = 0

                for mapNum, button in enumerate(buttons_with_map):
                    # print("Button clicked")
                    error_extracting = 0
                    teams = []
                    playersInMatch = []
                    if mapNum == 0:
                        active_table_xpath = (
                            "//div[contains(@id, 'panel-overview')]//table/tbody"
                        )
                    else:
                        active_table_xpath = f"//div[contains(@id, 'panel-game-{mapNum-1}')]//table/tbody"
                    table_body = WebDriverWait(driver, 3).until(
                        EC.presence_of_element_located(
                            (
                                By.XPATH,
                                active_table_xpath,
                            )
                        )
                    )

                    # print("Table found: ", table_body)

                    rows = table_body.find_elements(By.XPATH, "./tr")

                    for i, row in enumerate(rows):  # player rows
                        tds = row.find_elements(By.XPATH, "./td")
                        nameCol = tds[0].find_element(By.XPATH, "./a/div")
                        innerTxt = nameCol.get_attribute("innerText")
                        if i == 0 or i == 5:
                            # print("innertxt teamname: ", innerTxt)
                            teams.append(innerTxt)
                            continue
                        elif innerTxt == player:
                            if mapNum == 0:  # overall
                                # print("Button contains overall!")
                                mode = "Overall"
                                kills = row.find_element(
                                    By.XPATH, "./td[2]"
                                ).get_attribute("innerText")
                                deaths = row.find_element(
                                    By.XPATH, "./td[3]"
                                ).get_attribute("innerText")
                                kd = row.find_element(
                                    By.XPATH, "./td[4]"
                                ).get_attribute("innerText")
                                dmg = row.find_element(
                                    By.XPATH, "./td[6]"
                                ).get_attribute("innerText")
                                hillTime = None
                                firstBloods = None
                                ticks = None

                            elif mapNum == 1 or mapNum == 4:  # hardpoint
                                mode = "HardPoint"
                                kills = row.find_element(
                                    By.XPATH, "./td[2]"
                                ).get_attribute("innerText")
                                deaths = row.find_element(
                                    By.XPATH, "./td[3]"
                                ).get_attribute("innerText")
                                kd = row.find_element(
                                    By.XPATH, "./td[4]"
                                ).get_attribute("innerText")
                                dmg = row.find_element(
                                    By.XPATH, "./td[6]"
                                ).get_attribute("innerText")
                                hillTime = row.find_element(
                                    By.XPATH, "./td[7]"
                                ).get_attribute(
                                    "innerText"
                                )  # seconds
                                firstBloods = None
                                ticks = None
                            elif mapNum == 2 or mapNum == 5 or mapNum == 7:  # SnD
                                mode = "Search_and_Destroy"
                                kills = row.find_element(
                                    By.XPATH, "./td[2]"
                                ).get_attribute("innerText")
                                deaths = row.find_element(
                                    By.XPATH, "./td[3]"
                                ).get_attribute("innerText")
                                kd = row.find_element(
                                    By.XPATH, "./td[4]"
                                ).get_attribute("innerText")
                                dmg = row.find_element(
                                    By.XPATH, "./td[6]"
                                ).get_attribute("innerText")
                                firstBloods = row.find_element(
                                    By.XPATH, "./td[7]"
                                ).get_attribute(
                                    "innerText"
                                )  # seconds
                                hillTime = None
                                ticks = None
                            elif mapNum == 3 or mapNum == 6:  # Control
                                mode = "Control"
                                kills = tds[1].get_attribute("innerText")
                                deaths = tds[2].get_attribute("innerText")
                                kd = calculate_kd(int(kills), int(deaths))
                                dmg = tds[5].get_attribute("innerText")
                                ticks = tds[6].get_attribute("innerText")  # seconds
                                hillTime = None
                                firstBloods = None
                            else:
                                error_extracting = 1
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
                                # "Date": formatted_date,
                                "Kills": kills,
                                "Deaths": deaths,
                                "KD": kd,
                                "Damage": dmg,
                                "HillTime": hillTime,
                                "FirstBloods": firstBloods,
                                "Ticks": ticks,
                            }

                            mapName = "Overall" if mapNum == 0 else f"Map {mapNum}"
                            if error_extracting:
                                print(
                                    f"Error extracting data for {player} in match #{match_id} on {mapName}"
                                )
                            else:
                                successfulExtracts += 1
                                print(
                                    f"Successfully extracted data for {player} in match #{match_id} on {mapName}"
                                )
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
                totalMatchesSuccessfullyScraped += 1

            except Exception as e:
                print(f"Error extracting data for {player} in match #{match_id}:")
                print(e)
                continue
            print(
                f"Successfully extracted {successfulExtracts}/{len(buttons_with_map)} maps(including overview) for {player} in match #{match_id}"
            )
        print(
            f"Successfully extracted data on {totalMatchesSuccessfullyScraped}/{totalMacthesToScrape} matches for {player}"
        )

    # print(f"{player} : {match_links}")

    driver.quit()
    return allPlayerStats


def clean_data(data):
    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Convert Damage to numeric by removing commas
    df["Damage"] = df["Damage"].str.replace(",", "").astype(float)

    # Drop the Date column
    # df.drop(columns=["Date"], inplace=True)

    # Fill missing values with appropriate strategies
    df["HillTime"] = df["HillTime"].fillna(0)
    df["FirstBloods"] = df["FirstBloods"].fillna(0)
    df["Ticks"] = df["Ticks"].fillna(0)

    # Ensure numeric columns are of correct type
    numeric_columns = [
        "Kills",
        "Deaths",
        "KD",
        "Damage",
        "HillTime",
        "FirstBloods",
        "Ticks",
    ]
    df[numeric_columns] = df[numeric_columns].astype(float)

    return df


def preprocess_data_player_role_combinations(df):
    # One-hot encode categorical variables for modes and teams
    categorical_columns = ["Mode", "PlayerTeam", "EnemyTeam"]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=True)

    # Rename columns to avoid spaces
    df.columns = df.columns.str.replace(" ", "_")

    # Get unique player names from all relevant columns
    all_players = pd.Series(
        df[
            [
                "Player",
                "TeamMate1",
                "TeamMate2",
                "TeamMate3",
                "Enemy1",
                "Enemy2",
                "Enemy3",
                "Enemy4",
            ]
        ].values.ravel("K")
    ).unique()

    # Create binary features for each player-role combination
    for player in all_players:
        df[f"{player}_Player"] = df["Player"].apply(lambda x: 1 if x == player else 0)
        df[f"{player}_Teammate"] = df.apply(
            lambda row: (
                1
                if player in [row["TeamMate1"], row["TeamMate2"], row["TeamMate3"]]
                else 0
            ),
            axis=1,
        )
        df[f"{player}_Enemy"] = df.apply(
            lambda row: (
                1
                if player
                in [row["Enemy1"], row["Enemy2"], row["Enemy3"], row["Enemy4"]]
                else 0
            ),
            axis=1,
        )

    # Drop the original player-related columns
    df.drop(
        columns=[
            "Player",
            "TeamMate1",
            "TeamMate2",
            "TeamMate3",
            "Enemy1",
            "Enemy2",
            "Enemy3",
            "Enemy4",
        ],
        inplace=True,
    )

    # Scale numerical features
    numeric_columns = [
        "Kills",
        "Deaths",
        "KD",
        "Damage",
        "HillTime",
        "FirstBloods",
        "Ticks",
    ]
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df


# Example usage:
def main():
    allPlayerStats = scrape()  # Your scrape function here

    # Clean the data
    cleaned_data = clean_data(allPlayerStats)

    # Preprocess data using the chosen approach
    preprocessed_data = preprocess_data_player_role_combinations(cleaned_data.copy())

    # Save cleaned (but not scaled) data to CSV
    cleaned_data.to_csv("cleaned_player_stats.csv", index=False)

    # Save preprocessed (scaled) data to CSV
    preprocessed_data.to_csv("preprocessed_player_stats.csv", index=False)

    # Print some of the data to verify
    print(cleaned_data.head())
    print(preprocessed_data.head())


if __name__ == "__main__":
    main()
