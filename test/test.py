# Student agent: Add your own agent here
import random
from math import sqrt, log2
from copy import deepcopy
import numpy as np
import time


def check_endgame(board, my_pos, adv_pos):
    # checks if the node represents an end of game state
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    board_size = board.shape[0]
    # Union-Find
    father = dict()
    for r in range(board_size):
        for c in range(board_size):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2

    for r in range(board_size):
        for c in range(board_size):
            for dir, move in enumerate(moves[1:3]):  # Only check down and right
                if board[r, c, dir + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(board_size):
        for c in range(board_size):
            find((r, c))

    p0_r = find(tuple(my_pos))
    p1_r = find(tuple(adv_pos))
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)

    #returns true if the game is done
    return (p0_r != p1_r, p0_score, p1_score)


class MCTSNode:
    """
    a class representing a node of a monte carlo search tree
    the property state will represent the chess board
    """

    def __init__(self, parent=None, children: list = [], value: float = 0,
                 chess_board=None, my_pos: tuple = None, adv_pos: tuple = None):
        self.__parent = parent  # type MCTSNode as well
        self.__children = children
        self.__wins = 0
        self.__num_visits = 0
        self.__value = value
        self.board = chess_board  # deep copy needed?
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.board_size = self.board.shape[0]
        self.max_step = (self.board_size+1)//2

    def __str__(self):
        # for debugging purposes
        # can also add the chess board and its size later
        if self.__parent == None:
            p = None
        else:
            p = self.__parent.my_pos
        s0 = "###########################################################################################\n"
        s = f'MCTS Node with parent: {p}, children: {self.__children}, and {self.__wins} wins'
        s2 = f"\n my position {self.my_pos} and adversary position {self.adv_pos}\n"
        return s0 + s + s2

    def is_leaf(self):
        # check if the game is ended.
        return len(self.__children) == 0

    def has_parent(self):
        # check where at the monte carlo tree the node is placed.
        return self.__parent is not None

    def update_visits(self, n_visits: int):
        self.__num_visits += n_visits

    def add_node(self, board, my_pos, adv_pos):
        # add a child and update attributes
        new_node = MCTSNode(parent=self, children=[], chess_board=board, my_pos=my_pos, adv_pos=adv_pos)
        self.__children.append(new_node)

    def update_value(self, new_value: float):
        self.__value = new_value

    def get_children(self):
        return self.__children

    def get_value(self):
        return self.__value

    def get_parent(self):
        return self.__parent

    def get_visits(self):
        return self.__num_visits


single_ucb = lambda a, b, c, d: a + (d * sqrt(log2(b) / c))


class MCTS:
    """
    represents the tree search for the game
    uses MCTSNode as underlying data structure and follows the MCTS 4 processes to update
    the probabilities in each iteration
    """
    global single_ucb

    # deleted root from constructor because we already make a root
    def __init__(self, w: float = sqrt(2), chess_board=None, my_pos=None, adv_pos=None):
        self.root = MCTSNode(chess_board=chess_board, my_pos=my_pos, adv_pos=adv_pos)
        self.EXPLORATION_W = w  # TODO: exploration weight must be fine tuned in order to achieve the best result

    def UCB(self, children: list[MCTSNode]):
        # compute ucb using the global lambda function
        ucbs = [single_ucb(child.get_value(), child.get_parent().get_visits(), child.get_visits(), self.EXPLORATION_W)
                for child in children]
        max_index = max(range(len(ucbs)), key=lambda x: ucbs[x])
        return children[max_index]

    def random_move(self, root: MCTSNode):
        # randomly choose a next state for expansion phase
        # moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # steps = np.random.randint(0, root.max_step + 1)
        pass

    def _select(self) -> tuple:
        # based on the Upper Confidence Bound formula, use C as a tunable bias parameter
        root = self.root
        path = []
        while True:
            path.append(root)
            if root.is_leaf():
                return path, root
            # else: for all children, pick the max ucb value
            root = self.UCB(root.get_children())

        # find all possible valid moves BFS

    def _expand(self, root: MCTSNode):

        if check_endgame(root.board, root.my_pos, root.adv_pos)[0]:
            print("the game is ended")
            return root

        # find all possible valid moves BFS
        board_size = root.board_size

        # find allowed moves - BFS
        allowed = {}  # contains elements of type ((r, c): [list of dirs])
        queue = [root.my_pos]  # contains elements of type (r, c)
        path_lengths = {root.my_pos: 0}

        moves_dict = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}  # maps movement coords to values used in chess board
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        visited = []
        for i in range(board_size):
            visited.append([False for j in range(board_size)])

        #can also use the random agent move choice
        while queue != []:
            cur = queue.pop(0)
            r, c = cur
            visited[r][c] = True

            if path_lengths[cur] > root.max_step:
                break

            # if no border and is not the opponent's position
            for i in range(4):
                if not root.board[r, c, i] and cur != root.adv_pos:
                    if cur not in allowed:  # new position
                        allowed[cur] = [i]
                    else:
                        allowed[cur].append(i)
                    # print("added node in position", cur)

            # add all children of cur
            for m in moves:
                child = (r + m[0], c + m[1])  # child

                # check if new_pos is out of bounds AND if it was visited before
                if child[0] >= 0 and child[0] < board_size and child[1] >= 0 and child[1] < \
                        board_size and not visited[child[0]][child[1]] and child not in queue \
                        and not root.board[cur[0], cur[1], moves_dict[m]]:
                    queue.append(child)
                    # print("\tadded children in queue", child)
                    if child not in path_lengths:
                        path_lengths[child] = path_lengths[cur] + 1

        # print(len(allowed), "allowed moves")
        # print(allowed)

        # pick random valid move
        new_pos = random.choice(list(allowed))
        new_dir = random.choice(allowed[new_pos])  # need to take into account barrier as part of the move

        # update board
        new_board = deepcopy(root.board)  # need to keep old board for other moves
        new_board[new_pos[0], new_pos[1], new_dir] = True
        # pick the symmetric move
        if new_dir == 0:
            new_board[new_pos[0] - 1, new_pos[1], 2] = True
        elif new_dir == 1:  # double check this condition
            new_board[new_pos[0], new_pos[1] + 1, 3] = True
        elif new_dir == 2:
            new_board[new_pos[0] + 1, new_pos[1], 0] = True
        elif new_dir == 3:
            new_board[new_pos[0], new_pos[1] - 1, 1] = True


        # TODO: do we have to change the board or the positions - basically do we have to make a move?
        # or is that done when simulating?
        # we have to add all possible children
        root.add_node(new_board, new_pos, root.adv_pos)
        print("expanded node position", root.get_children()[-1].my_pos, "with direction", new_dir)
        #print(root.get_children()[-1].board)
        return root.get_children()[-1]

    def _simulate(self, root: MCTSNode):
        '''
        Runs one random simulation of the game starting from the chosen action node
        :return: True if win, False if loss/tie
        '''

        chess_board_copy = deepcopy(root.board) #don't want to modify the original board in root
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        agent_turn = False  # start with opponent's turn

        player_pos = deepcopy(root.my_pos)
        other_pos = deepcopy(root.adv_pos)
        end, my_score, op_score = check_endgame(chess_board_copy, player_pos, other_pos)  # tuple (the game ended, my_score, opponent score)

        while not end:

            #swap original positions
            player_pos, other_pos = other_pos, player_pos
            ori_pos = deepcopy(player_pos)

            # generate random move
            steps = random.randint(0, root.max_step)

            # Random Walk
            for _ in range(steps):
                r, c = player_pos
                dir = random.randint(0, 3)
                m_r, m_c = moves[dir]
                player_pos = (r + m_r, c + m_c)

                # Special Case enclosed by Adversary
                k = 0
                #check if border blocks you from taking next step
                while chess_board_copy[r, c, dir] or player_pos == other_pos:
                    k += 1
                    if k > 300:
                        break
                    dir = random.randint(0, 3)
                    m_r, m_c = moves[dir]
                    player_pos = (r + m_r, c + m_c)

                if k > 300:
                    player_pos = ori_pos  # don't move them - nowhere to go
                    break

            # Put Barrier
            dir = random.randint(0, 3)
            r, c = player_pos
            while chess_board_copy[r, c, dir]:
                dir = random.randint(0, 3)

            # move current player to chosen position
            chess_board_copy[player_pos[0], player_pos[1], dir] = True  # place barrier
            # pick the symmetric move
            if dir == 0:
                chess_board_copy[player_pos[0] - 1, player_pos[1], 2] = True
            elif dir == 1:  # double check this condition
                chess_board_copy[player_pos[0], player_pos[1] + 1, 3] = True
            elif dir == 2:
                chess_board_copy[player_pos[0] + 1, player_pos[1], 0] = True
            elif dir == 3:
                chess_board_copy[player_pos[0], player_pos[1] - 1, 1] = True

            if agent_turn:
                print("My turn:", "my position:", player_pos, "with direction", dir, "opponent position:", other_pos)
                end, my_score, op_score = check_endgame(chess_board_copy, player_pos, other_pos)

            else:
                print("Opponent's turn:", "my position:", other_pos, "opponent position:", player_pos, "with direction",
                      dir)
                end, my_score, op_score = check_endgame(chess_board_copy, other_pos, player_pos)

            # change whose turn it is
            agent_turn = not agent_turn
            #print(chess_board_copy)
            #print("________________________________________")

        #the score comparison is wrong
        return my_score > op_score

    def _run_simulations(self, root: MCTSNode, n: int = 1):
        # runs n simulations from the root node
        # returns the average win rate
        n_wins = 0

        for i in range(n):
            print("_________________________________")
            if self._simulate(root):
                n_wins += 1

        return n_wins

    def _backpropagate(self, path: list, n_wins: int, visits: int):
        # updates the values and visits of each node along the path
        for node in path:
            print("backpropagating")
            print("printing the path size")
            print(len(path))
            # update value
            prev_wins = node.get_visits() * node.get_value()
            new_value = (prev_wins + n_wins) / (node.get_visits() + visits)
            node.update_value(new_value)

            node.update_visits(10)  # make sure this is what we want

def step(self, chess_board, my_pos, adv_pos, max_step):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (x_max, y_max, 4)
    - my_pos: a tuple of (x, y)
    - adv_pos: a tuple of (x, y)
    - max_step: an integer

    You should return a tuple of ((x, y), dir),
    where (x, y) is the next position of your agent and dir is the direction of the wall
    you want to put on.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """
    # MCTS algorithm
    # make root
    r = MCTSNode(chess_board=chess_board, my_pos=my_pos, adv_po=adv_pos, max_step=max_step)
    mcts = MCTS(root=r, chess_board=self.chess_board, my_pos=my_pos, adv_pos=adv_pos)

    # 1. selection
    path, root = mcts._select()

    # 2. expansion at the last node on path
    expanded_node = mcts._expand(path[-1])

    # 3. simulation
    # need to finetune number of simulations
    runs = 1
    n_wins = mcts._run_simulations(expanded_node, runs)

    # 4. backpropagation
    mcts._backpropagate(path, n_wins, runs)

    return my_pos, dir


# Testing
if __name__ == '__main__':
    # initialize board
    board_size = 7
    chess_board = np.zeros((board_size, board_size, 4), dtype=bool)

    # Set borders, empty board
    chess_board[0, :, 0] = True
    chess_board[:, 0, 3] = True
    chess_board[-1, :, 2] = True
    chess_board[:, -1, 1] = True

    my_pos = (3, 3)
    adv_pos = (1, 1)

    # extra borders
    chess_board[4, 0, 1] = True
    chess_board[4, 1, 3] = True
    chess_board[3, 1, 1] = True
    chess_board[3, 2, 3] = True
    chess_board[6, 3, 0] = True
    chess_board[5, 3, 2] = True
    chess_board[0, 3, 2] = True
    chess_board[1, 3, 0] = True
    chess_board[1, 4, 2] = True
    chess_board[2, 4, 0] = True
    chess_board[2, 4, 1] = True
    chess_board[2, 5, 3] = True



    # Maximum Steps
    max_step = (board_size + 1) // 2

    random.seed(7)
    mcts = MCTS(chess_board=chess_board, my_pos=my_pos, adv_pos=adv_pos)

    for i in range(10):
        # 1. selection
        path, root = mcts._select()
        print("root: ", root)
        '''
        print("PATH: ")
        for n in path:
            print(n)
        '''

        # 2. expansion at the last node on path
        expanded_node = mcts._expand(path[-1])
        # print(expanded_node.get_children())
        #expansion chances the board in a weird way
        print("chosen node: ", expanded_node)

        #print(expanded_node.board == chess_board)


        # 3. simulation
        # need to finetune number of simulations

        #some infinite loop happens here
        now = time.time()
        runs = 500

        n_wins = mcts._run_simulations(expanded_node, runs)
        then = time.time()
        print(f'the time complexity = {then - now}')
        print("wins:", n_wins)

        # 4. backpropagation
        expanded_node.update_visits(runs)
        mcts._backpropagate(path, n_wins, runs)
        for n in path:
            print(n)
            print("printing the value")
            print(n.get_value())





