import json
from pprint import pprint
import random

output_folder = '/Users/aakaashkapoor/Desktop/Data_271/changed/'
input_folder = '/Users/aakaashkapoor/Desktop/Data_271/original/'
filename = 'apache__abdera.json'
FILE_CHANGE_PROBABLITY = 0.3

# function that reads in a file and reads the json
def read_json_from_file(filename):
    with open(input_folder + filename) as json_data:
        d = json.load(json_data)
        json_data.close()
        return d




# function that deletes random nodes from the tree and makes the relevant changes to file
def make_function_changes(json_data):

    # From next token
    to_delete_next_token = random.randint(0, len(json_data["Edges"]["NextToken"]) - 1)
    del json_data["Edges"]["NextToken"][to_delete_next_token]

    # From last lexical use
    to_delete_last_lexical = random.randint(0, len(json_data["Edges"]["LastLexicalUse"]) - 1)
    del json_data["Edges"]["LastLexicalUse"][to_delete_last_lexical]

    # From child
    to_delete_child = random.randint(0, len(json_data["Edges"]["Child"]) - 1)
    del json_data["Edges"]["Child"][to_delete_child]

    # printing
    print("Function:", json_data["MethodName"], "| Deleted Child:", to_delete_child, "| Deleted last lexical:", to_delete_last_lexical, "| Deleted next token:", to_delete_next_token)

    return json_data

# save data to file
def save_json_data(new_data, filename):

    with open(output_folder + filename, 'w') as json_file:
        json.dump(new_data, json_file)

# make changes in a file and creates a new file
def create_new_file(filename):

    data = read_json_from_file(filename)
    new_data = []

    for func in data:

        episilon = random.random()

        # for only changing 30% files
        if(episilon < FILE_CHANGE_PROBABLITY):

            # modify function in data
            new_func = make_function_changes(func)

            new_func["IsChanged"] = True
        else:
            print("Not Changing Function")

            # added is changed property
            new_func = func
            new_func["IsChanged"] = False

        # add to the new data list
        new_data.append(new_func)

    save_json_data(new_data, filename)

create_new_file(filename)






# pprint(d)