import argparse
parser = argparse.ArgumentParser()
parser.add_argument("square", type=int,
                    help="计算数字的平方")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="可视化")
args = parser.parse_args()
answer = args.square**2
if args.verbose:
    print(f"the square of {args.square} equals {answer}")
else:
    print(answer)
