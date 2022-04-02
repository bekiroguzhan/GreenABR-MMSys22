import pathlib
import os 
import sys 


DIR_NAME=sys.argv[1]
update=sys.argv[2]

def main():

        for path in pathlib.Path(DIR_NAME).glob("**/*"):
            if path.is_file():
                fname = path.stem
                print(fname, "old")
                old_extension = path.suffix
                if old_extension=='.csv':
                        directory = path.parent
                        n_fname=str(fname)
                        n_fname=n_fname+'_'+str(update)+str(old_extension)
                        print('new',n_fname)
                        path.rename(pathlib.Path(directory, n_fname))

if __name__=='__main__':
        main()