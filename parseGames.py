import chess.pgn
import os
import multiprocessing
import h5py
import numpy as np
from random import choice

def getInputFiles(files=[], folder="./Games"):

    for fileIn in os.listdir(folder): #iterates through all game files in 'Games'
        if fileIn.endswith(".pgn"): #if file already converted to HDF5, skip
            fileIn = os.path.join(folder, fileIn) #define full path for file 
            fileOut = fileIn.replace(".pgn", ".hdf5") #convert file from pgn to HDF5
            if not os.path.exists(fileOut): #Check file doesn't already exist
                files.append((fileIn, fileOut)) #add input/output files to list of files

    print(files)
    #pool = multiprocessing.Pool() #Pools the available processors/cores, meaning they can process the games in parallel
    for filePair in files:
        readAllGames(filePair) #Parses and processes the contents of the given .pgn file, writing it's ouput to the paired .hdf5 file 
        print(f"Finished reading: {filePair}") #Indicates when the last file finished being parsed
    


def readAllGames(files):    
    fileIn, fileOut = files #unpack input and output games 
    with h5py.File(fileOut, "w") as gameDataset: #create HDF5 object to initialise the datasets for this game
        Xp, Xq, Xr = [gameDataset.create_dataset(datasetName, (0, 64), dtype="b", maxshape=(None, 64), chunks=True) for datasetName in ["xp", "xq", "xr"]] #initialise subgroups (datasets) for positions (p,q,r)
        Y, M = [gameDataset.create_dataset(datasetName, (0,), dtype="b", maxshape=(None,), chunks=True) for datasetName in ["y", "m"]] #initialise subgroups for moves till game end (M) and game result (Y)
                                               #signed byte     #has ~unlimited size   #dataset stored as chunked storage, to allow resizing in the future
        
        size = 0 #max number of lines which can be written
        line = 0 #denotes the number of lines in the datasets which have been written to
        for gameSequence in readGames(fileIn): #iterate through the returned generator for move sequences
            print("#####################")
            print(f"Reading game #{line}...")
            print("#####################")
            game = parseGame(gameSequence) #returns the flattened boards, moves until the game ends from said boards, and the game result
            if game is None: #discard training example if the game wasnt over in the final move
                print("\nSkipping training example...\n")
                continue
            x, xParent, xRandom, movesLeft, y = game 

            if line + 1 >= size: #dataset has reached it's maximum size
                gameDataset.flush() #flush the hdf5 buffers, to prevent the buffered data being written to disk (and exceeding the max size)
                size = 2 * size + 1 #increase max size of dataset 
                print(f"Resizing to size: {size}")
                [gameDataset[datasetName].resize(size, axis=0) for datasetName in ("xp", "xr", "xq", "y", "m")]

            #update all datasets with the parsed boards/data
            Xq[line] = x 
            Xr[line] = xRandom
            Xp[line] = xParent
            Y[line] = y
            M[line] = movesLeft

            line += 1 #each game is 1 line in the dataset

        [gameDataset[datasetName].resize(line, axis=0) for datasetName in ("xp", "xr", "xq", "y", "m")] #resizes datasets, releasing unused storage (line denotes the number of lines which have been written) 
    
def parseGame(game):
    endConditions = {"1-0": 1, "1/2-1/2": 0, "0-1": -1} #maps end-game conditions from the move sequences
    result = game.headers["Result"] #access game result from game headers 
    if result not in endConditions: #means a player timed out, so we discard training example
        return None
    y = endConditions[result] #store translated game result in y

    endNode = game.end() #fetches the last node in the move sequence
    if not endNode.board().is_game_over(): #if in the final board state the game isn't over, discard training example
        return None

    nodes = [] #tracks information on nodes(moves) in the game, starting from the end node
    movesLeft = 0 #moves to get from current node to end node
    while endNode: #until all moves are parsed
        nodes.append((movesLeft, endNode, endNode.board().turn)) #'turn' returns True if white to move and False if black to move (in the current node)
        endNode = endNode.parent #fetches the node above the current node (nodes are stored in a tree)
        movesLeft += 1 #since current node is now 1 more move above the end node

    nodes.pop() #remove node representing board with no moves made

    movesLeft, endNode, whiteMove = choice(nodes) #randomly chooses node from game

    board = endNode.board() #get the board from the random node
    x = baseBoardToArray(board) #converts the board into a flattened 64-element array of squares 
    boardParent = endNode.parent.board() #get the board from before the move was made 
    xParent = baseBoardToArray(boardParent) #converts the previous board into flattened array
    if not whiteMove:
        y = -y #negate game result, so the reuslt for a won game is constant (1) for either side

    # generate a random baord
    moves = list(boardParent.legal_moves) #gets list of all the possible moves which could have been made from the previous board variation
    move = choice(moves) #returns randomly selected move from list of legal moves
    boardParent.push(move) #makes the (random) move, which is reflected on the board
    xRandom = baseBoardToArray(boardParent) #converts this 'random' board into a flattened array

    return (x, xParent, xRandom, movesLeft, y) #returns the flattened boards, moves until the game ends, and the game result

def readGames(file):
    with open(file) as gameFile:
        while True: 
            try:
                game = chess.pgn.read_game(gameFile) #reads the game (move) sequence from file
            except KeyboardInterrupt: #stop executution if Ctrl-c pressed
                raise

            if not game: #if all games read from file, stop reading
                break
        
            yield game #return genreator, so program doesn't have to parse all games at once


def baseBoardToArray(board): #converts the standard chess board into a flattened array
    boardArray = np.zeros(64, dtype=np.int8) #array of 64 8-bit integers, squares which aren't occupied will be left as 0
    for square in range(64): #iterate through the 64 squares on board
        piece = board.piece_type_at(square) #returns the integer representing the piece at given square (e.g. ROOK=4)
        if piece: #piece will be None if square was empty
            colour = int(bool(board.occupied_co[chess.BLACK] & chess.BB_SQUARES[square])) #occupied_co returns a 64-bit integer mask of the squares occupied by the given player, BB_squares returns the integer mask for the passed square
                                                                                      #Using bitwise 'and' (&), it can be deduced which player occupies the given square
            col = square % 8 #convert mask to board indexes 
            row = square // 8
            piece = (colour*7) + piece #used to distinguish between white and black pieces. Max value of original piece_type is 6, so we add 7 if it's a white piece.

            boardArray[row * 8 + col] = piece #We insert values from the same row in the same sequence of 8 indexes - which uses the fact that rows are 8 squares 'apart'

    return boardArray





if __name__ == "__main__":
    getInputFiles()