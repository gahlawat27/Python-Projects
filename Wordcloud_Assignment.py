#!/usr/bin/env python
# coding: utf-8

# In[24]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re


# # Part 1: Fetching Webpage Content

# In[25]:


url = "http://quotes.toscrape.com"
response = requests.get(url)

if response.status_code == 200:
    html_content = response.content
else:
    print(f"Error fetching the webpage. Status code: {response.status_code}")
    exit()


# # Part 2: Parsing HTML Content

# In[26]:


soup = BeautifulSoup(html_content, "html.parser")

quotes = []
authors = []

quote_elements = soup.find_all("div", class_="quote")
for quote_element in quote_elements:
    quote_text = quote_element.find("span", class_="text").get_text(strip=True)
    author_element = quote_element.find("small", class_="author")
    author_name = author_element.get_text(strip=True)
    quotes.append(quote_text)
    authors.append(author_name)


# # Part 3: Saving Scraped Data

# In[27]:


data = {"Quote": quotes, "Author": authors}
df = pd.DataFrame(data)
df.to_csv("quotes.csv", index=False)


#  # Part 3: Data Visualization - Word Cloud

# In[28]:


text = " ".join(quotes)
text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
text = text.lower()  # Convert to lowercase


# In[29]:


# Display the word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white")
wordcloud.generate(text)


# In[30]:


# Save the word cloud as an image
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Terms")
plt.savefig("word_cloud.png")
plt.show()


# In[ ]:





# In[ ]:




