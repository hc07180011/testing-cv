from frames.process import process

files = [ "002", "003", "006",
        "016", "044", "055",
        "070", "108", "121",
        "169"]

def _main():
    for f in files:
        process(f)

if __name__ == "__main__":
    _main()