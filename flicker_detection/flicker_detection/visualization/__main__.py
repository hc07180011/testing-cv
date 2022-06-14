import sys, getopt
from visualization.pca import pca
from visualization.kernel_pca import kernel_pca

def _main(argv):
    input_directory = ''
    try:
        opts, args = getopt.getopt(argv, 'hi:a:', ['ifile=', 'algorithm='])
    except getopt.GetoptError:
        print ('Usage: -i <inputdirectory> -a <algorithm>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-i <inputdirectory>')
            sys.exit()
        elif opt in('-i', '--ifile'):
            input_directory = arg
        elif opt in('-a', '--algorithm'):
            alg = arg
    if (input_directory):
        if (alg == 'kernel_pca'):
            print('Kernel PCA')
            kernel_pca(input_directory)
        else:
            print('PCA')
            pca(input_directory)
            
if __name__ == '__main__':
    _main(sys.argv[1:])
