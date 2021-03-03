from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.keys import Keys

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import pandas as pd
import datetime
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


class Additional_Song_Data_Scraper():
    def __init__(self):
        self.percent_labels = ["Acousticness", "Energy", "Liveness", "Speachiness", "Daneability", "Instrumentalness", "Loudness", "Valence"]# do not change
        self.driver = webdriver.Chrome(ChromeDriverManager().install())
        self.basic_search_url = "https://songdata.io/search"
        self.wait_seconds = 5
        self.song_data_list = []

    def search_song(self, search_string, song_id):

        first_retry = True

        # Initialize Driver
        self.driver.get(self.basic_search_url)

        # Search for the given search string
        WebDriverWait(self.driver, self.wait_seconds).until(
            EC.element_to_be_clickable((By.XPATH, '//*[contains(concat( " ", @class, " " ), concat( " ", "fa-search", " " ))]')))

        self.driver.find_element_by_xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "form-control", " " ))]').send_keys(search_string) # Search Field
        self.driver.find_element_by_xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "fa-search", " " ))]').click() # Search Button

        # Click on the first found result
        try:
            WebDriverWait(self.driver, self.wait_seconds).until(
                EC.element_to_be_clickable((By.XPATH, '//tr[(((count(preceding-sibling::*) + 1) = 1) and parent::*)]//a'))).click() # First found result
        except TimeoutException:
            print("exception")
            if first_retry:
                self.search_song(search_string, song_id)
            else:
                return

        # Find and scrap Key, Camlelot, BPM, Length, Release Data, Time Signature and Loudness
        WebDriverWait(self.driver, self.wait_seconds).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#loudness")))
        basic_infos = [element.text for element in self.driver.find_elements_by_css_selector('#loudness .text-white+ .text-white , #time .text-white+ .text-white , .display-4.mb-0 , .display-4.mb-0 , #release .text-white+ .text-white , #length .text-white+ .text-white')]

        # Receive the given percentages of the Song Analysis
        percentages = [int(element.text[:-1]) for element in self.driver.find_elements_by_css_selector('span')[3:]]
        url = self.driver.current_url

        # Song_id, Key, Camlelot, BPM, Length, Release Data, Time Signature, Loudness, 8x Song Analytics, Url
        current_song_data = [song_id] + basic_infos + percentages + [url]
        self.song_data_list.append(current_song_data)

    def create_dataframe(self):
        self.df = pd.DataFrame(columns=[
            "song_id", "key", "camelot", "bpm", "length", "release_date", "time_signature", "loudness_decibel",
            "acousticness", "energy", "liveness", "speachiness", "danceability", "instrumentalness", "loudness", "valence", "url"],
        data=self.song_data_list)

        self.df["loudness_decibel"] = self.df["loudness_decibel"].apply(lambda x: float(x[:-3])) # remove "db" at the end and convert to float
        self.df["length"] = self.df["length"].apply(lambda x: int(x.split(":")[0])*60 + int(x.split(":")[1])) # Convert 3:31 into 211 seconds
        self.df["release_date"] = self.df["release_date"].apply(lambda x: datetime.datetime.strptime(x, "%B %d, %Y")) # Convert "3" into 3

        self.df["time_signature"] = self.df["time_signature"].apply(lambda x: int(x)) # Convert "3" into 3
        self.df["bpm"] = self.df["bpm"].apply(lambda x: int(x)) # Convert "3" into 3

        return self.df

    def save_dataframe(self):
        self.df.to_csv(path_or_buf = "additional_song_data.csv", index=False, encoding = "utf-8")


        '''
        # Print the scraped information
        print(basic_infos)
        for label, per in zip(self.percent_labels, percentages):
            print(f"{label}: {per}")

        print(url)'''


ass = Additional_Song_Data_Scraper()
ass.search_song("Britney Oops", 10)
ass.search_song("Aqua Barbie Girl", 512)
df = ass.create_dataframe()
ass.save_dataframe()





# Close the driver session
ass.driver.close()

