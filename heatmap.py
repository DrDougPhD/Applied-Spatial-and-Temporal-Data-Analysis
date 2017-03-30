import collections
import pprint
import matplotlib.pyplot as plt


users = set()
restaurants = set()
print('Preprocessing and counting')
with open('data/restaurant_ratings.txt') as f:
    for line in f:
        user, restaurant, rating, timestamp = [int(v) for v in line.split()]
        users.add(user)
        restaurants.add(restaurant)

num_users = len(users)
num_restaurants = len(restaurants)
user_ratings = [
    [0 for r in range(max(restaurants)+1)]
    for u in range(max(users)+1)
]
print('Building matrix')
with open('data/restaurant_ratings.txt') as f:
    for line in f:
        user, restaurant, rating, timestamp = [int(v) for v in line.split()]
        user_ratings[user][restaurant] = rating

print('Drawing image')
plt.pcolor(user_ratings)
plt.colorbar()
plt.title('User Ratings')
plt.xlabel('Restaurants')
plt.ylabel('Users')
plt.savefig('heatmap.png')

"""
plt.title('Restaurant Ratings for each User')
plt.ylabel('Restaurant Rating')
plt.xlabel('User')
plt.savefig('heatmap.svg')
"""

