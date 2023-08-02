import datetime
from typing import Any, Tuple

from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt


class Player:
    """
    The player data class

    Attributes
    --------------------
    _playertime: float
        The playtime of the player on the server
    _money: float
        The amount of money the player owns
    _player_level: int
        The RPG level of the player
    _start_time: int
        The timestamp of the first time the player on the server
    """
    def __init__(self, playtime: float, money: float, player_level: float, start_time: float):
        self._playtime = playtime
        """ The playtime of the player on the server """
        self._money = money
        """ The amount of money the player owns """
        self._player_level = player_level
        """ The RPG level of the player """
        self._start_time = start_time
        """ The timestamp of the first time the player on the server """

    def normalise(self, range_stats: tuple[Any, Any, Any, Any, Any, Any, Any, Any]):
        """
        Normalizes the player's data using Z-Score

        :param range_stats: The list of the means and standard deviations of the player's stats
        """

        # Means and Standard Deviations for each of the player's stats
        # Playertime
        # Money
        # Player Level
        # Start Date
        pt_mean, pt_stdev, \
            m_mean, m_stdev, \
            lvl_mean, lvl_stdev, \
            st_mean, st_stdev = range_stats

        # Normalises the attributes
        self._playtime = (self._playtime - pt_mean) / pt_stdev
        self._money = (self._money - m_mean) / m_stdev
        self._player_level = (self._player_level - lvl_mean) / lvl_stdev
        self._start_time = (self._start_time - st_mean) / st_stdev

    def denormalise(self, range_stats: tuple[Any, Any, Any, Any, Any, Any, Any, Any]):
        """
        Denormalizes the player's data from their Z-Score

        :param range_stats: The list of the means and standard deviations of the player's stats
        """

        # Means and Standard Deviations for each of the player's stats
        # Playertime
        # Money
        # Player Level
        # Start Date
        pt_mean, pt_stdev, \
            m_mean, m_stdev, \
            lvl_mean, lvl_stdev, \
            st_mean, st_stdev = range_stats

        # Denormalises the attributes
        self._playtime = self._playtime * pt_stdev + pt_mean
        self._money = self._money * m_stdev + m_mean
        self._player_level = self._player_level * lvl_stdev + lvl_mean
        self._start_time = self._start_time * st_stdev + st_mean

    def get_array(self) -> np.ndarray[float]:
        """
        :return: The numpy array of the player's stats
        """
        return np.array([self._playtime,
                        self._money,
                        self._player_level,
                         self._start_time])


def get_player_level(player_data: list[float]) -> int:
    """
    :param player_data: The record of the player's data from the csv file
    :return: The calculated player level from the player's stats
    """
    total_stats = 0

    # The stats are the last 5 entries in the record
    for i in player_data[-6:]:
        total_stats += i

    # (Sum(Stats) - 55) / 5
    return (total_stats - 55) / 5


def generate_player_list() -> list[Player]:
    """
    Generates a list of the players from the csv file

    :return: The player dataset
    """
    player_list = []

    with open("data.csv", 'r') as f:
        # Skips the headers line
        f.readline()

        while (line := f.readline()) != "":
            player_data = [float(x) for x in line.replace("\n", "").split(",")]

            player_list.append(Player(player_data[2],player_data[3], get_player_level(player_data),
                                      player_data[0]))

    return player_list


def get_range_stats(player_list: list[Player]) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """
    :param player_list: The player dataset
    :return: Means and Standard Deviations for the player's stats
        (Playertime,
        Money,
        Player Level,
        Start Date)
    """
    player_array_list = [x.get_array() for x in player_list]

    means = np.mean(player_array_list, axis=0)
    stdevs = np.std(player_array_list, axis=0)

    return means[0], stdevs[0], \
            means[1], stdevs[1], \
            means[2], stdevs[2], \
            means[3], stdevs[3], \



class Kmeans:
    """
    Implementation of the K-Means clustering algorithm

    Attributions
    -------------------
    _groups: list[list[Player]]
        The groups for the current iteration of the algorithm
    _data: list[Player]
        The player dataset
    _group_means: list[Player]
        The centres of each cluster, represented as a "Mean Player"
    """
    def __init__(self, groups: int, data: list[Player]):
        self._groups = [[] for _ in range(groups)]
        """ The groups for the current iteration of the algorithm """
        self._data = data
        """ The player dataset """

        np.random.shuffle(self._data)
        self._group_means = np.random.choice(data, groups)
        """ The centres of each cluster, represented as a "Mean Player" """

    def run(self, iterations=300) -> tuple[list[list[Player]], list[Player]]:
        """
        Generates the K-Means clusters

        :param iterations: The amount of iterations the algorithm will run for
        :return: The clusters and the centres of the clusters
        """
        for i in range(iterations):
            print(i)

            self._add_items()
            self._select_group_means()

        return self._groups, self._group_means

    def _select_group_means(self):
        """
        Regenerates the list of cluster centres from the current groups
        """
        mean_data = [np.mean([p.get_array() for p in group], axis=0) for group in self._groups]

        self._group_means = []

        for i in range(len(mean_data)):
            self._group_means.append(Player(
                mean_data[i][0],
                mean_data[i][1],
                mean_data[i][2],
                mean_data[i][3],
            ))

    def _add_items(self):
        """
        Adds the player dataset to the clusters each player is closest to
        """
        self._groups = [[self._group_means[i]] for i in range(len(self._group_means))]

        for datapoint in self._data:
            index = self._get_closest_index(datapoint)
            self._groups[index].append(datapoint)

    def _get_closest_index(self, p: Player) -> int:
        """
        :param p: The player being added to a cluster
        :return: The index of the cluster the player is closest to, -1 if error
        """
        cur_index = -1
        cur_dist = float("inf")

        for i in range(len(self._group_means)):
            mean_point = self._group_means[i]
            dist = self._dist(p, mean_point)

            if dist < cur_dist:
                cur_index = i
                cur_dist = dist

        return cur_index

    def _dist(self, p1: Player, p2: Player):
        """
        :param p1:
        :param p2:
        :return: The Euclidean distance between the stats of the two players
        """
        d = np.sum((p1.get_array() - p2.get_array())**2)

        return np.sqrt(d)


def db_index(clusters: list[list[Player]], cluster_means: list[Player]) -> float:
    """
    :param clusters:
    :param cluster_means:
    :return: The Davies–Bouldin index for the clusters generated
    """
    average_score = 0

    average_intras = []

    for i in range(len(clusters)):
        avg_intra = 0

        for j in range(len(clusters[i])):
            avg_intra += np.sum(np.sqrt((clusters[i][j].get_array() - cluster_means[i].get_array()) ** 2))

        average_intras.append(avg_intra / len(clusters[i]))

    for i in range(len(clusters)):
        max_r = -float("inf")

        for j in range(len(clusters)):
            if i != j:
                r = (average_intras[i] + average_intras[j]) \
                        / np.sum(np.sqrt((cluster_means[i].get_array() - cluster_means[j].get_array()) ** 2))

                max_r = max(r, max_r)

        average_score += max_r

    return average_score / len(clusters)


def silhouette_eval(groups: list[list[Player]], means: list[Player]):
    """
    :param groups:
    :param means:
    :return: The Silhouette score for the clusters
    """
    s = 0

    for k in range(len(groups)):
        group = groups[k]
        for i in range(len(group)):
            a = 0
            b = float("inf")

            p1_matrix = group[i].get_array()
            for j in range(len(group)):
                if i != j:
                    p2_matrix = group[j].get_array()

                    dist = np.sum((p1_matrix - p2_matrix) ** 2)

                    a += dist

            a /= len(group)

            for j in range(len(groups)):
                if j != k:
                    p2_matrix = means[j].get_array()
                    dist = np.sum((p1_matrix - p2_matrix) ** 2)

                    b = min(b, dist)

            s += (b - a) / max(a, b)

    return s / sum([len(group) for group in groups])


def generate_median_players(groups):
    """
    :param groups:
    :return: The median players for each cluster
    """
    median_data = [np.median([p.get_array() for p in group], axis=0) for group in groups]
    medians = []

    for i in range(len(median_data)):
        medians.append(Player(
            median_data[i][0],
            median_data[i][1],
            median_data[i][2],
            median_data[i][3],
        ))

    return medians


def order_groups(groups: list[list[Player]], medians: list[Player]) \
        -> tuple[list[list[Player]], list[Player]]:
    """
    :param groups: The clusters for the player dataset
    :param medians: The median players for each cluster
    :return: The ordered groups and medians in terms of the median player's start time
    """
    median_start_times = [player._start_time for player in medians]
    sorted_start_times = median_start_times.copy()

    sorted_start_times.sort()
    st_indices = [median_start_times.index(x) for x in sorted_start_times]

    return [groups[x] for x in st_indices], [medians[x] for x in st_indices]


def generate_definining_traits_graphs(group_names: list[str],
                                      medians: list[Player]):
    """
    Generates a graph demonstrating the graphs defining attributes, which are used in the
    clustering algorithm

    :param group_names: The names of each group
    :param median_start_times: The start times of each median player
    :param medians: The median players of each cluster
    """
    fig, ax = plt.subplots(2, 2)

    fig.suptitle("Group Defining Traits")

    # Sets the titles for each sub-graph
    ax[0, 0].set_title("Playtime Hours per Group")
    ax[0, 1].set_title("Player Level per Group")
    ax[1, 0].set_title("Money per Group")
    ax[1, 1].set_title("Starting Date per Group")

    # Only shows the bottom spine of the graph
    ax[0, 0].spines[['top', 'left', 'right']].set_visible(False)
    ax[0, 1].spines[['top', 'left', 'right']].set_visible(False)
    ax[1, 0].spines[['top', 'left', 'right']].set_visible(False)
    ax[1, 1].spines[['top', 'left', 'right']].set_visible(False)

    # Sets the x-axis labels for the first 3 graphs as the clusters' names
    ax[0, 0].set_xticks([x for x in range(4)], group_names)
    ax[0, 1].set_xticks([x for x in range(4)], group_names)
    ax[1, 0].set_xticks([x for x in range(4)], group_names)

    # Removes the y-axis (de-cluttering)
    ax[0, 0].set_yticks([], [])
    ax[0, 1].set_yticks([], [])
    ax[1, 0].set_yticks([], [])
    ax[1, 1].set_yticks([], [])

    # Sets the bounds of the date graph between 2 and -2
    # This allows for the bars to alternate above and below the x-axis
    # for each date, with enough room for the group name labels
    ax[1, 1].set_ylim(-2, 2)

    # The median player start times
    median_start_times = [player._start_time for player in medians]
    median_start_times.append(datetime.datetime.now().timestamp() * 1000)

    # Gets the range of the start times, for normalisation
    st_min = np.min(median_start_times)
    st_max = np.max(median_start_times)

    norm_start_times = [(x - st_min) / (st_max - st_min) for x in median_start_times]

    # Formats the start times as Year/Month/Day
    start_times_dates = [datetime.datetime.fromtimestamp(x / 1000). \
                             strftime("%Y/%m/%d") for x in median_start_times]


    # Sets the x-axis as the normalised start times
    ax[1, 1].set_xticks(norm_start_times, start_times_dates)

    # Sets the bottom spine as the x-ais
    ax[1, 1].spines['bottom'].set_position('zero')

    # Generates the charts
    for group_index in range(len(playtime)):
        # Creates the bar charts
        ax[0, 0].bar(group_index, (medians[group_index]._playtime // (20 * 3600)))
        ax[0, 1].bar(group_index, medians[group_index]._player_level)
        ax[1, 0].bar(group_index, medians[group_index]._money)
        ax[1, 1].bar(norm_start_times[group_index], (-1) ** group_index, width=5e-3)

        # Labels the bar charts
        ax[0, 0].text(group_index, (medians[group_index]._playtime // (20 * 3600)) * 0.75,
                      format((medians[group_index]._playtime // (20 * 3600)), ','),
                      ha="center", fontweight="bold")
        ax[0, 1].text(group_index, medians[group_index]._player_level * 0.75,
                      format(int(medians[group_index]._player_level), ','),
                      ha="center", fontweight="bold")
        ax[1, 0].text(group_index, medians[group_index]._money * 0.75,
                      format(int(medians[group_index]._money), ','),
                      ha="center", fontweight="bold")
        ax[1, 1].text(norm_start_times[group_index], (-1) ** group_index * 1.1,
                      group_names[group_index],
                      ha="center", fontweight="bold")


def generate_player_level_pdf(player_levels: list[list[float]]):
    """
    Generates charts of the player levels as a probablity distribution function

    :param player_levels: The levels for the player, grouped by the clusters
    """
    fig, axes = plt.subplots(2, 2)

    fig.suptitle("Player Level per Group")

    for i in range(len(player_levels)):
        index_0 = i // 2
        index_1 = i % 2

        # Creates 15 bins for the pdf
        bins = 15

        # Gets the x coordinates needed to plot a bell curve over the pdf
        func_x_values = np.linspace(np.min(player_levels[i]), np.max(player_levels[i]))

        # Formats the x-axis to be more reader friendly
        level_x_ticks = np.linspace(np.min(player_levels[i]), np.max(player_levels[i]), num=6)
        level_x_ticks_formatted = [format(int(x / 1e3),',') + "K" for x in level_x_ticks]

        axes[index_0, index_1].set_xticks(level_x_ticks, level_x_ticks_formatted)

        # Removes the y-axis and the top, left, and right spines
        axes[index_0, index_1].set_yticks([], [])
        axes[index_0, index_1].spines[['top', 'left', 'right']].set_visible(False)

        # Sets the title for each sub-graph as the group name
        axes[index_0, index_1].set_title(group_names[i])

        # Creates a histogram for the pdf
        axes[index_0, index_1].hist(player_levels[i], bins=bins, density=True)

        # Plots a bell curve using the group's mean and standard deviation
        axes[index_0, index_1].plot(func_x_values,
                                    norm.pdf(func_x_values, np.mean(player_levels[i]), np.std(player_levels[i])),
                                    color="black")


def generate_money_pdf(moneys: list[float]):
    """
    Generates charts of the players money as a probablity distribution function

    :param moneys: The money for the player, grouped by the clusters
    """

    fig, axes = plt.subplots(2, 2)

    fig.suptitle("Money per Group")

    for i in range(len(moneys)):
        index_0 = i // 2
        index_1 = i % 2

        # Creates 15 bins for the pdf
        bins = 15

        # Gets the x coordinates needed to plot a bell curve over the pdf
        func_x_values = np.linspace(np.min(moneys[i]), np.max(moneys[i]))

        # Formats the x-axis to be more reader friendly
        money_x_ticks = np.linspace(np.min(moneys[i]), np.max(moneys[i]), num=6)
        money_x_ticks_formatted = [format(int(x / 1e3),',') + "K" for x in money_x_ticks]

        axes[index_0, index_1].set_xticks(money_x_ticks, money_x_ticks_formatted)

        # Removes the y-axis and the top, left, and right spines
        axes[index_0, index_1].set_yticks([], [])
        axes[index_0, index_1].spines[['top', 'left', 'right']].set_visible(False)

        # Sets the title for each sub-graph as the group name
        axes[index_0, index_1].set_title(group_names[i])

        # Creates a histogram for the pdf
        axes[index_0, index_1].hist(moneys[i], bins=bins, density=True)

        # Plots a bell curve using the group's mean and standard deviation
        axes[index_0, index_1].plot(func_x_values,
                                    norm.pdf(func_x_values, np.mean(moneys[i]), np.std(moneys[i])),
                                    color="black")


# Loads the player data
player_data = generate_player_list()


# Normalises the player's stats
range_stats = get_range_stats(player_data)
for player in player_data:
    player.normalise(range_stats)

# Runs the K Means algorithm
kmeans = Kmeans(4, player_data)
groups, means = kmeans.run(iterations=100)

# Evaluates the generated clusters
db_score = db_index(groups, means)
print(f"D.B. Index: {db_score}")

# Denormalises the players' data
for group in groups:
    for player in group:
        player.denormalise(range_stats)

for player in means:
    player.denormalise(range_stats)

# Generates the median players, and orders the groups in terms of start time
medians = generate_median_players(groups)
groups, medians = order_groups(groups, medians)

# Group names
group_names = ["Ancients", "Infrequents", "New Wealth", "Newbies"]

# Individual player stats per group
start_times = [[player._start_time for player in group] for group in groups]
playtime = [[player._playtime for player in group] for group in groups]
player_levels = [[player._player_level for player in group] for group in groups]
moneys = [[player._money for player in group] for group in groups]

# Graphs
generate_definining_traits_graphs(group_names, medians)
generate_player_level_pdf(player_levels)
generate_money_pdf(moneys)

plt.show()
