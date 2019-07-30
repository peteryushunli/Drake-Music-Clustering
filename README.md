# Drake Music Clustering
 ![Source: The Ringer](https://cdn-images-1.medium.com/max/800/1*VA3zi4trqMG_9Y5tR0QrmA.png)
 
# Introduction
Aubrey "Drake" Graham is one of today's biggest music superstars. The Canadian rapper-singer dominates both the pop charts and internet meme culture. In the decade since his critically acclaimed mixtape, So Far Gone, Drake has released 7 solo projects and 1 joint mixtape on his way to becoming the most streamed artist of all time.
In an industry full of one-hit wonders and changing fads, what has allowed this former teen-drama actor to stay atop the game? The answer is versatility. Drake's biggest strength is that he can seamlessly transition from hip-hop bravado to moody ballads that strike the perfect balance between sensitivity and relatability. From UK grime to dancehall to reggaeton, he's shown that there's no sound he can't adapt to. While many will argue over the authenticity of his approach, its success is undisputed.
To further analyze his versatility and musical evolution I set out to use machine learning to identify changing patterns within his music and create clusters (groupings) based on metrics of each song.
# Data Source
The primary dataset was obtained via Spotify's Web API containing musical attributes from Echo Nest that are defined below:
Acousticness: Confidence measurement of whether the track is acoustic
Danceability: How danceable a song based on musical elements including tempo, rhythm stability, beat strength, and overall regularity
Energy: A perceptual measure of intensity and activity. Features that contribute to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy
Instrumentalness: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental, while rap or spoken word are clearly "vocal". Values above 0.5 are intended to represent instrumentals
Loudness: Overall loudness in decibels (dB) averaged across the entire track. Useful for comparing relative loudness of tracks
Speechiness: The presence of spoken words in a track. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music.
Additional lyrical data was web-scraped using a python package created by johnwmillr leveraging the web API of popular lyrics website, Genius.
The final dataset includes 162 Drake songs found on Spotify. These do not include any songs where he was featured on another artist's song.
![Preview of the data](https://cdn-images-1.medium.com/max/1200/1*IPfGO8sc4oJL8wbOhPHlfA.png)
# Exploring the data behind Drake's music
![Correlation Heatmap](https://cdn-images-1.medium.com/max/800/1*jFP9ZvTumSRpUqTcdHGKcA.png)
The correlation heatmap shows a correlation score (color) between every data pointBased on the correlation heatmap above, we see a correlation where songs are growing in popularity and danceability over time. There is also a negative correlation where songs are becoming shorter in length (minutes) over time. This suggests that Drake's music over time has grown shorter and more rhythmic, and overall more radio friendly.
Next, I will deep-dive into these attributes and examine how these data points reflect Drake's stylistic changes in the latter half of his career.
## 1. Nothing Was the Same…after 2015
![Cat-plot](https://cdn-images-1.medium.com/max/600/1*KEcvV4FrTp-9vnBOSZV9Ow.png) ![Box-plot](https://cdn-images-1.medium.com/max/600/1*T46Ce032WA6yBKmswtE1vg.png)

The cat-plot shows the danceability scores of each Drake's songs, and the box-plot shows the length of each song over time. The plots above show Drake's music becoming shorter and more danceable over time, which points to a general shift towards pop. This is also due to the rise of streaming applications like Spotify, which incentivize shorter songs since revenue is generated per stream.
2015 is also when both of these metrics shift significantly. This a pivotal year in Drake's career; he released two mixtapes, a viral diss record (Back to Back), and his most important single to date, Hotline Bling.
Despite its lack of marketing - released on SoundCloud as a b-side filler to his original Meek Mill diss track, Charged Up (the prequel to Back to Back) - Hotline Bling peaked at 2nd place on the Billboards Hot 100 (his highest mark since his 2009 debut single, Best I Ever Had). The song would epitomize Drake's transformation into the internet superstar he is today through genre-bending sounds and a music video filled with unabashed corniness and countless dank memes:
![Memes](https://cdn-images-1.medium.com/max/800/1*FrielnkqGAwwLUllhMLoCA.gif)
## 2. Effect on Lyricism
![Box-plot](https://cdn-images-1.medium.com/max/600/1*pVvVT4NAGbmuBkUJXLXk1w.png)

(Note: the edge of the box represent 1st and 3rd quartile, middle line is the median and whiskers represent min and max, all diamonds are considered outliers)As his songs and overall brand became more mainstream, I wanted to explore how this affected his lyricism as a rapper. To measure this, I used unique word count as the metric to gauge creativity and effort in song writing (influenced by this article on rappers' vocabulary, and this Machine Learning NLP article on Drake).
The result here is a bit unclear, but I would argue there is a slight negative trend from 2013 onwards. The increase we see in 2017 is likely due to More Life being a compilation mixtape that featured many different artists. Overall though, I would say his lyricism has been largely unaffected. Rather than writing songs with fewer unique words, he found a sweet spot of between 50 to 55 unique words as evidenced by his 2018 double-album, Scorpion, which had the most songs of any project (25) but smallest variance in word count.
# Clustering Drake's Music
Clustering is a form of unsupervised machine learning that "identifies commonalities in the data and reacts based on the presence or absence of such commonalities". In other words, it groups data points together through observed patterns. Since our dataset contains primarily numerical attributes, machine learning can automatically create clusters of songs
![Example](https://cdn-images-1.medium.com/max/800/1*X8gDNl-I9Lcj_48CsYVtyA.jpeg)
## How many clusters do we create?
![Elbow Plot](https://cdn-images-1.medium.com/max/800/1*co48otl5fFRIrbx5DAqaAw.png)

One common way to determine the optimal number of clusters is to use the elbow method, which looks for an "elbow" in the plot where adding an additional cluster (x-axis) results in minimal decrease in distortion (y-axis).
The blue line measures distortion based on clusters (k), and the dotted green line measures timeThe elbow plot for our data actually shows the ideal number of clusters being 3. However given Drake's famed versatility and the scope of this analysis, it would be very boring to only have 3 groups of Drake songs. Instead, I went with the next ideal number, which was 5.
## What metrics do we use?
I used a standard k-means clustering model on the following numerical data attributes: minutes, popularity, danceability, acousticness, energy, instrumentalness, liveness, loudness, speechiness, tempo and unique word count. I dropped year from the model because it has no inherent significance in distinguishing the musical element of a song. The aggregated results for each cluster are below:
![Cluster Preview](https://cdn-images-1.medium.com/max/1200/1*TOyLdYTdNqOqcZ4Fg2fCAg.png)

Now let's dive into each cluster!
## Cluster 0: "The Come Up"
*Money just changed everything, I wonder how life without it would go*
![Cluster List](https://github.com/yushunli2013/yushunli2013.github.io/blob/master/images/Project%20Images/Drake%20Clustering/cluster%200%20list.png)


Not the flashiest or most popular aspect of Drake's arsenal, but this collection of songs show his comeup as an artist with introspective topics about chasing his dreams, fleeting happiness and dealing with adversity.
Song Attributes:
* Low tempo and danceability reflect most of the songs' introspective mood and message with few exceptions (Over, Fancy, HYFR)
* High unique word count show a focus on lyricism and storytelling

## Cluster 1: "Versatile Hit Maker"
*She say, "Do you love me?" I tell her "Only partly", I only love my bed and my momma, I'm sorry*
![Cluster List](https://github.com/yushunli2013/yushunli2013.github.io/blob/master/images/Project%20Images/Drake%20Clustering/cluster%201%20list.png)


Drake's biggest cluster both in terms sheer number of songs, but also in terms of versatility and number of hit records. Stylistically there is no true common theme other than they're all very catchy, easy to listen to and popular.
Song Attributes:
* High tempo and danceability with the lowest unique word count (average of 8 fewer words per song) among all clusters show that songs are more melodic, catchy and radio-friendly
* Highest instrumentalness reflect a diverse range of production and sub-genres including trap, bounce, dancehall, etc.
Contains most of his most popular hits like God's Plan, Nice For What, Passionfruit, Started From the Bottom, Fake Love, One Dance

## Cluster 2: "Leader of the New School"
* Drinking every night because we drink to my accomplishments*
![Cluster List](https://github.com/yushunli2013/yushunli2013.github.io/blob/master/images/Project%20Images/Drake%20Clustering/cluster%202%20list.png)


Stylistically very similar to cluster 0, but these hip-hop songs display more bravado than introspection as Drake works to cement his legacy as the biggest and most successful rapper of his generation, and possibly all-time.
Song Attributes:
* Unliked Cluster 0, these songs have the highest tempo among all clusters making for fast-paced rap songs with intricate rhyme patterns and delivery
* Drake's message here is more braggadocios and focuses on money and status. However, these themes are primarily used to show his status within hip-hop and to standout among his peers
* Many songs are from his sophomore classic, Take Care, including his early hits, Headlines

## Cluster 3: "Trap Star"
*Got a sneaker deal and I ain't break a sweat*
![Cluster List](https://github.com/yushunli2013/yushunli2013.github.io/blob/master/images/Project%20Images/Drake%20Clustering/cluster%203%20list.png)


This is the most genre-specific cluster captures the popular sub-genre of trap music. In particular, this cluster shows Drizzy's ability to adapt and change with musical trends by adopting this Atlanta influenced sound of heavy 808-drums, hi-hats and triplet flows.
Song Attributes:
* Unique combination of high danceability, tempo and speechiness; mostly fast-paced, catchy rap songs best suited for parties and nightlife
* Contains many similar features from Atlanta rappers like Future, 2Chainz, Quavo and Young Thug

## Cluster 4: "In My Feelings"
*I'm just saying you can do better, tell me have you heard that lately?*
![Cluster List](https://github.com/yushunli2013/yushunli2013.github.io/blob/master/images/Project%20Images/Drake%20Clustering/cluster%204%20list.png)

This cluster of songs best exemplify his signature style of moody R&B mixed with emotional rapping on top of muddy low-end bass drums. A popular style that's influenced newer artists like Bryson Tiller, Amir Obe, 6lack and more.
Song Attributes:
* Majority of theses songs are produced by his long-time music partner, 40; this signature moody, slow sound provides the low energy and tempo, while Drake's melancholy vocals make up the low speechiness
* Thematically, the song focus around breakups and unrequited love, best exemplified by his classic R&B track, Marvin's Room

# Model Conclusion
* Overall the clusters fully captured 5 unique sounds within Drake's vast discography, albeit with some songs in each cluster that were misplaced
* Cluster 1 lacks clear musical or lyrical theme. There are many songs that dispersed into other clusters for better accuracy (this is could be due to the popularity metric being similar among these songs)
* Despite removing years from the clustering analysis, clusters still generally show songs from similar time periods/albums. This means that the musical and thematic shifts made over time are reflected through the data

# Final Thoughts and Next Steps
This analysis not only provided an in-depth look into Drake's musical evolution, but also showed the power of machine learning. A simple clustering model could accurately cluster a broad set of songs, and serve as the foundation to create personalized playlists and recommendations. Streaming services like Spotify are surely creating much, much more sophisticated models and playlists based on these types of data and more.
As someone deeply passionate about hip-hop, I want to continue using data from Spotify's Web API to explore other rappers like Kendrick Lamar and J.Cole, and compare their music against Drake or other pop artists to better understand how they differ both musically through data.
