import requests
from bs4 import BeautifulSoup
import copy

# URL of the website you want to scrape
url = 'http://www.cubesolv.es/'

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content of the page
soup = BeautifulSoup(response.content, 'html.parser')

# Find elements by their HTML tags, classes, ids, etc.
# For example, to find all <a> tags (links) on the page:

# Print the URLs of all links found on the page
for j in range(1, 5960):
    response = requests.get(url + f"/solve/{j}")

    soup = BeautifulSoup(response.content, 'html.parser')

    h2_tags = soup.find_all('h2')

    h2_tags_text_arr = [tag.get_text() for tag in h2_tags]

    if(len(h2_tags_text_arr) == 0):
        continue

    print(j)

    if(h2_tags_text_arr[0].find("3x3") == -1):
        print("No 3x3 found")
    else:
        if(len(soup.find_all('div', class_="algorithm well")) < 2):
            print("No solution found")
            continue
        scramble = soup.find_all('div', class_="algorithm well")[0].get_text()
        scramble = scramble.strip()

        solution = soup.find_all('div', class_="algorithm well")[1].get_text()
        solution = solution.strip()
        solution = solution.split("\n")
        solution = [element.strip() for element in solution]
        solution_temp = copy.deepcopy(solution)
        solution = []
        for element in solution_temp:
            if element != "":
                solution.append(element)

        solution_str = ""
        i = 0
        while i < len(solution):
            if(solution[i] == "//"):
                solution_str += "//"
                solution_str += solution[i+1] + "\n"
                i += 2
            else:
                solution_str += solution[i] + " "
                i+=1

        # Write the scramble and solution to a text file
        with open('scramble_solution.txt', 'a') as file:
            file.write("Scramble: " + scramble + "\n")
            file.write("Solution: " + solution_str + "\n\n")
