from cgitb import text
from google.protobuf import text_format
import multiclass.new_label_pb2 as nl2

def relabel(fpath):
    f = open(fpath, "r")
    new_labels = nl2.Labels()
    message = text_format.MessageToString(f.read(), new_labels)
    # new_labels.ParseFromString()
    f.close()