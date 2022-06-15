import sys, getopt
from hamming.hamming import hamming_distance

def _main(argv):
    image1 = ''
    image2 = ''
    alg = ''
    try:
        opts, args = getopt.getopt(argv, 'h', ['file1=', 'file2=', 'algorithm='])
    except getopt.GetoptError:
        print ('Usage: --file1 <image1> --file2 <image2>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('--file1 <image1> --file2 <image2>')
            sys.exit()
        elif opt == '--file1':
            image1 = arg
        elif opt == '--file2':
            image2 = arg
        elif opt == '--algorithm':
            alg = arg
    
    if (image1 and image2):
        print('Hamming distance: ' + str(hamming_distance(image1, image2, alg)))
            
if __name__ == '__main__':
    _main(sys.argv[1:])
