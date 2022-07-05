from multiclass.relabel import relabel

fpath = './multiclass/new_label.textproto'

def _main():
    relabel(fpath)

if __name__ == "__main__":
    _main()