import argparse
parser = argparse.ArgumentParser()
#parser.add_argument("echo" , help="打印字符串")
#args = parser.parse_args()
#print(args.echo)
# 需要在终端跑
# argparse是python标准库中用于命令行解析的模块

#<1>
parser.add_argument("square" , help="打印一个数字的平方" , type=int)
args = parser.parse_args() # 这里的parse_args()是从选项中返回具体的数据
print(args.square**2)
