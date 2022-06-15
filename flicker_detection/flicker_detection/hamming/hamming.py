from PIL import Image
import imagehash

def hamming_distance(image1, image2, alg):
    """
    Calculate the hamming distance between two images.
    """
    if (alg == 'average_hash' or alg == ''):
        image1Hash = imagehash.average_hash(Image.open(image1))
        image2Hash = imagehash.average_hash(Image.open(image2))
    elif (alg == 'phash'):
        image1Hash = imagehash.phash(Image.open(image1))
        image2Hash = imagehash.phash(Image.open(image2))
    elif (alg == 'dhash'):
        image1Hash = imagehash.dhash(Image.open(image1))
        image2Hash = imagehash.dhash(Image.open(image2))
    # elif (alg == 'whash'):
    #     image1Hash = imagehash.whash(Image.open(image1))
    #     image2Hash = imagehash.whash(Image.open(image2))
    elif (alg == 'colorhash'):
        image1Hash = imagehash.colorhash(Image.open(image1))
        image2Hash = imagehash.colorhash(Image.open(image2))
        
    return image1Hash - image2Hash
    
