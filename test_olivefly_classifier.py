import glob
import sys
import argparse
import pathlib
import cv2

parser = argparse.ArgumentParser(
    prog="OliveFly detection test script",
    description="""
        This scripts tests the olive fly detection algorithm
        against a set of images.""")

parser.add_argument('directory', help='location of the dataset',
                    type=pathlib.Path)
parser.add_argument('--verbose', '-v', action="store_true")


def detect_olive_fly(image) :
    # TODO: implement something here 
    return False

def main():
    args = parser.parse_args()
    
    # confusion matrix
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for filename in glob.glob(str(args.directory)+"/**/*.JPG", recursive=True) :
        img = cv2.imread(filename)
        if "not_olive_fly" in filename :
            olive_fly = False
        elif "olive_fly" in filename:
            olive_fly = True
        else :
            print (f"{filename} not labeled.")
            continue

        detection_result = detect_olive_fly(img)

        if olive_fly and detection_result :
            TP += 1
        elif olive_fly and not detection_result :
            FN += 1
        elif not olive_fly and detection_result :
            FP += 1
        else :
            TN += 1
        if args.verbose :    
            if detect_olive_fly(img) :
                print (f"{filename} contains an olive fly.")
            else:
                print (f"{filename} does not contain an olive fly.")

    print (f"Summary: True positives {TP}, False positives {FP}")
    print (f"Summary: false negatives {FN}, True negatives {TN}")
    return 0

if __name__ == '__main__' :
    sys.exit(main())
    
    
        
