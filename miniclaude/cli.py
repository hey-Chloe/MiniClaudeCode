import argparse
from miniclaude.agent import Agent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('task')
    args=parser.parse_args()

    result=Agent().run(args.task)

    for item in result:
        print(item)
