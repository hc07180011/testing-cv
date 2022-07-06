import json

fpath = './data/label.json'

if __name__ == '__main__':
    with open(fpath) as f:
        data = json.load(f)

        for d in data:
            print(data[d])