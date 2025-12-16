import os
import re
import pandas as pd

EMPTY_LINES = 14

starting_folder = "Data"

spectral_data = {}

# For pandas dataframe
spectral_data_app = {}
index = 0

# Read all folder inside the starting folder
for date_acquisition in os.listdir(starting_folder):
    # Check if the folder is a directory
    if os.path.isdir(os.path.join(starting_folder, date_acquisition)):
        # Read all acquisition folder inside the data acquisition folder
        for n_acquisition in os.listdir(os.path.join(starting_folder, date_acquisition)):
            # Check if folder is a directory
            if os.path.isdir(os.path.join(starting_folder, date_acquisition, n_acquisition)):
                # find the first match of one or more digits (\d+) in the input string
                number = int(re.findall(r'\d+', n_acquisition)[0])
                # Read all files inside the acquisition folder
                for file in os.listdir(os.path.join(starting_folder, date_acquisition, n_acquisition)):
                    # Check if file is a .txt file
                    if file.endswith(".txt"):
                        # print(os.path.join(starting_folder, date_acquisition, n_acquisition, file))
                        # Read the position of the lattuce and the acquisition number
                        match = re.search(r'([A-Za-z])_?(\d+).*Relative_*([0-9]*)', file)
                        position = match.group(1) + match.group(2)
                        n = int(match.group(3))
                        # Check if the position is already in the dictionary
                        if position not in spectral_data:
                            spectral_data[position] = {}
                        if date_acquisition not in spectral_data[position]:
                            spectral_data[position][date_acquisition] = {}
                        spectral_data[position][date_acquisition][n] = {}
                        # Read the file
                        with open(os.path.join(starting_folder, date_acquisition, n_acquisition, file), "r") as f:
                            # avoid the first 14 lines of the file
                            for _ in range(EMPTY_LINES):
                                f.readline()
                            # start reading the file
                            for line in f:
                                # from line read only the reflectance value
                                s = line.split()
                                wavelength = float(s[0].replace(",","."))
                                reflectance = float(s[1].replace(",","."))
                                spectral_data[position][date_acquisition][n][wavelength] = reflectance
                                
                                spectral_data_app[index] = {"Date": date_acquisition, "Acquisition": number, "N": n, "Position": position, "Wavelength": wavelength, "Reflectance": reflectance}
                                index += 1


# Read the wavelength, which will be the x axis                      
x = []
with open(os.path.join(starting_folder, date_acquisition, n_acquisition, file), "r") as f:
    # avoid the first 14 lines of the file
    for _ in range(EMPTY_LINES):
        f.readline()
    # start reading the file
    for line in f:
        # from line read only the reflectance value
        wavelength = float(line.split()[0].replace(",","."))
        x.append(wavelength)

df = pd.DataFrame.from_dict(spectral_data_app, "index")

# Add the class to the dataframe
df.loc[df.Date == 'day_5', 'Class'] = 'Unhealty'  
df.loc[df.Date == 'day_4', 'Class'] = 'Unhealty'  
df.loc[df.Date == 'day_3', 'Class'] = 'Unhealty'  
df.loc[df.Date == 'day_2', 'Class'] = 'Healty' 
df.loc[df.Date == 'day_1', 'Class'] = 'Healty' 

# Add the day to the dataframe
df.loc[df.Date == 'day_5', 'Day'] = '24-02-02'
df.loc[df.Date == 'day_4', 'Day'] = '24-02-03'
df.loc[df.Date == 'day_3', 'Day'] = '24-01-31'
df.loc[df.Date == 'day_2', 'Day'] = 'day_2'
df.loc[df.Date == 'day_1', 'Day'] = '24-01-29'

df.to_csv('spectral_data.csv',index=False)
    