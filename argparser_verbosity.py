import argparse
parser = argparse.ArgumentParser()
#parser.add_argument("--verbosity", help="increase output verbosity")

#parser.add_argument("--verbosity", help="increase output verbosity" , action="store_true")
# 添加的action的意思是当--verbosity选项不存在时候就是False,反之就是True

parser.add_argument("-v" , "--verbosity", help="increase output verbosity" , action="store_true")
args = parser.parse_args()
if args.verbosity:
    print("verbosity turned on")
