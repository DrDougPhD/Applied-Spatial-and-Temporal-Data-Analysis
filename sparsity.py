import collections
import pprint
import matplotlib.pyplot as plt


users = collections.defaultdict(set)
restaurants = set()
with open('data/restaurant_ratings.txt') as f:
    for line in f:
        user, restaurant, rating, timestamp = [int(v) for v in line.split()]
        users[user].add(restaurant)
        restaurants.add(restaurant)

print('Number of users:', len(users))
print('Number of restaurants:', len(restaurants))

num_ratings_for_all = [len(rated_restaurants)
                       for rated_restaurants in users.values()]
sorted_num_rating = sorted(num_ratings_for_all, reverse=True)
print('Highest ratings:', sorted_num_rating[0])
print('Lowest  ratings:', sorted_num_rating[-1])

plt.plot(sorted_num_rating)
plt.title('Number of Ratings per User')
plt.ylabel('Number of Rated Restaurants')
plt.xlabel('Ranked User')
plt.savefig('ratings_per_user.svg')
