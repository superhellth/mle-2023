from agents import Agent
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def print_data(agents):
    agent_list = []
    for agent in agents:
        agent_dict = {
            "agent_name": agent.get_state()[0],
        }
        agent_dict.update(agent.statistics)
        agent_list.append(agent_dict)
    print(agent_list)
    agent_names = [agent["agent_name"] for agent in agent_list ]
    try:
        update_json_file("test_statistics.json", agent_list)
        print("Combination exists. Updating...")
    except FileExistsError:
        print("Combination does not exist.")
    return 0



def update_json_file(json_file, new_data):
    try:
        if os.path.exists(json_file):
            with open(json_file, 'r') as file:
                try:
                    data = json.load(file)
                except json.decoder.JSONDecodeError:
                    data = []
        else:
            data = []

        data.append(new_data)

        with open(json_file, 'w') as file:
            json.dump(data, file, indent=4)
    except FileNotFoundError:
        pass

def calculate_average_stats(data):
    agent_stats = {}

    for run in data:
        for agent_data in run:
            agent_name = agent_data["agent_name"]
            if agent_name not in agent_stats:
                agent_stats[agent_name] = {key: 0 for key in agent_data.keys() if key != "agent_name"}

            for key, value in agent_data.items():
                if key != "agent_name":
                    if key in agent_stats[agent_name]:
                        agent_stats[agent_name][key] += value


    average_stats = {}

    for agent_name, stats in agent_stats.items():
        average_stats[agent_name] = {}
        for key, total_value in stats.items():
            average_value = total_value / len(data)
            average_stats[agent_name][key] = average_value

    return average_stats



# Example usage
with open("test_statistics.json", 'r') as file:
    try:
        data = json.load(file)
    except FileNotFoundError:
        data=[]
        print("Did not work")

average_stats = calculate_average_stats(data)
attributes = average_stats['ql_agent'].keys()
agents = average_stats.keys()
average_stats["akins_agent"]["invalid"] = 0.0
average_stats["subfield_agent"]["invalid"] = 0.0

penguin_means = {}

for attr in attributes:
    penguin_means[attr] = tuple(average_stats[agent][attr] for agent in agents)


x = np.arange(len(agents))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Length (mm)')
ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, agents)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 250)

plt.show()