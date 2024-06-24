from trees.TreeGenerator import TreeGenerator
from environment.Grid import Grid

def main():
    tree = TreeGenerator()
    treeOne = tree.generateTree("Abies Holophylla", (1, 2))
    grid = Grid(3, 3)
    grid.plant(1, 2, treeOne)



if __name__ == "__main__":
    main()
