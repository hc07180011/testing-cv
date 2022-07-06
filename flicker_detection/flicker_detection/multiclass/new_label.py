import json
import matplotlib.pyplot as plt

fpath = './data/new_label.json'

# Dictionary should look like:
# {
#   "fname": {
#       "frames_list": [[1, 2, 3], [5, 6]]
#       "class": [3, 2]
#   }
# }

if __name__ == '__main__':
    with open(fpath) as f:
        data = json.load(f)
        result_dict = {}
        n_elements = []
        for i in range(250):
            n_elements.append(0)

        for d in data:
            data_list = data[d]
            new_entry = {"frames_list": [], "class": []}
            if len(data_list):
                prev = data_list[0] - 1
                sequence = 0
                start_index = 0
                for i, element in enumerate(data_list):
                    if (element == prev + 1):
                        sequence += 1
                    else:
                        sliced_list = data_list[start_index: i]
                        start_index = i
                        new_entry["frames_list"].append(sliced_list)
                        new_entry["class"].append(sequence)
                        n_elements[sequence] += 1
                        sequence = 1
                    prev = element
                sliced_list = data_list[start_index:]
                new_entry["frames_list"].append(sliced_list)
                new_entry["class"].append(sequence)
                n_elements[sequence] += 1
            result_dict[d] = new_entry
    
    with open("multiclass.json", "w") as f:
        json.dump(result_dict, f)
    
    print(n_elements)
    plt.hist(n_elements, bins = len(n_elements))
    plt.savefig("result.png")
