import sys, getopt
from visualization.pca import pca
import tensorflow as tf

def _main(argv):
    input_directory = ''
    try:
        opts, args = getopt.getopt(argv, 'hi:', ['ifile='])
    except getopt.GetoptError:
        print ('Usage: -i <inputdirectory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-i <inputdirectory>')
            sys.exit()
        elif opt in('-i', '--ifile'):
            input_directory = arg
    if (input_directory):
        pca(input_directory)

if __name__ == '__main__':
    _main(sys.argv[1:])
