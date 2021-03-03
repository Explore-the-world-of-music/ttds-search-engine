from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.keys import Keys

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import pandas as pd
import datetime
import os
import time

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


class Additional_Song_Data_Scraper():
    def __init__(self):
        #self.percent_labels = ["Acousticness", "Energy", "Liveness", "Speachiness", "Daneability", "Instrumentalness", "Loudness", "Valence"]# do not change
        self.driver = webdriver.Chrome(ChromeDriverManager().install())
        self.basic_search_url = "https://songdata.io/search"
        self.wait_seconds = 1
        self.song_data_list = []
        self.unresolved_exceptions = 0

    def search_song(self, search_string, song_id, first_try = True):
        '''
        This function searches for a string on the songdata.io and appends the results to self.song_data_list

        :param search_string - string to search for on songdata.io (str)
        :param song_id - id with which the new data get saved (int)
        :param first_try - indicator for the recursive retry if information retrieval failed (bool)
        '''

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


            # Find and scrap Key, Camlelot, BPM, Length, Release Data, Time Signature and Loudness
            WebDriverWait(self.driver, self.wait_seconds).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#loudness")))
            basic_infos = [element.text for element in self.driver.find_elements_by_css_selector('#loudness .text-white+ .text-white , #time .text-white+ .text-white , .display-4.mb-0 , .display-4.mb-0 , #release .text-white+ .text-white , #length .text-white+ .text-white')]

            # Receive the given percentages of the Song Analysis
            percentages = [int(element.text[:-1]) for element in self.driver.find_elements_by_css_selector('span')[3:]]
            url = self.driver.current_url

            # Song_id, Key, Camlelot, BPM, Length, Release Data, Time Signature, Loudness, 8x Song Analytics, Url
            current_song_data = [song_id] + basic_infos + percentages + [url]
            self.song_data_list.append(current_song_data)
        except:
            #print(f"Exception for {song_id}")
            if first_try:
                self.search_song(search_string, song_id, first_try = False)
            else:
                self.unresolved_exceptions += 1
                return

    def date_converter(self, date):
        '''
        Function which tries to convert the date string into a datetime format

        :param date - date from the scraping (str)
        :return date - converted date (datetime)
        '''

        try:
            return datetime.datetime.strptime(date, "%B %d, %Y")
        except:
            return datetime.datetime.strptime(date, "%Y")


        

    def create_dataframe(self):
        '''
        Creates a dataframe with matching columns for all values in the self.song_data_list
        '''

        df = pd.DataFrame(columns=[
            "song_id", "key", "camelot", "bpm", "length", "release_date", "time_signature", "loudness_decibel",
            "acousticness", "energy", "liveness", "speachiness", "danceability", "instrumentalness", "loudness", "valence", "url"],
        data=self.song_data_list)

        df["loudness_decibel"] = df["loudness_decibel"].apply(lambda x: float(x[:-3])) # remove "db" at the end and convert to float
        df["length"] = df["length"].apply(lambda x: int(x.split(":")[0])*60 + int(x.split(":")[1])) # Convert 3:31 into 211 seconds
        df["release_date"] = df["release_date"].apply(lambda x: self.date_converter(x)) # Convert "3" into 3

        df["time_signature"] = df["time_signature"].apply(lambda x: int(x)) # Convert "3" into 3
        df["bpm"] = df["bpm"].apply(lambda x: int(x)) # Convert "3" into 3

        return df

    def save_new_data(self):
        '''
        Loads the existing csv, appends the new data, saves the df and clears the self.song_data_list
        '''
        df = pd.read_csv("additional_song_data.csv", encoding = "utf-8").append(self.create_dataframe(), ignore_index = True)
        df.to_csv(path_or_buf = "additional_song_data.csv", index=False, encoding = "utf-8")
        self.song_data_list = []


        '''
        # Print the scraped information
        print(basic_infos)
        for label, per in zip(self.percent_labels, percentages):
            print(f"{label}: {per}")

        print(url)'''


ass = Additional_Song_Data_Scraper()
#ass.search_song("Britney Oops", 10)
#ass.search_song("Aqua Barbie Girl", 512)
#df = ass.create_dataframe()
#ass.save_dataframe()

data = pd.read_csv("data-song_v1.csv")
data = data[0:50]
modulo = 10
currently_unresolved = 0

begin = time.time()
for ids, (idx, row) in enumerate(data.iterrows()):
    if ids%modulo == 0 and ids != 0:
        print(f"Saving Version {ids/modulo}", end = "\r")
        ass.save_new_data()
        print(f"v{ids/modulo}: {ids}/{data.shape[0]} !{ass.unresolved_exceptions-currently_unresolved}/{ass.unresolved_exceptions} - time for {modulo} songs: {time.time() - begin}")
        currently_unresolved = ass.unresolved_exceptions
        begin = time.time()

    ass.search_song(row["SongTitle"] + " " + row["ArtistMain"], row["SongID"])

ass.save_new_data()


# Close the driver session
ass.driver.close()

