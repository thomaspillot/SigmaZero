import pygame 
import os
import numpy as np
from pygame import init
import sys
import math
from train import MyModel
import random
import tensorflow as tf

pygame.font.init() #initialises pygame font child class
pygame.init() #initiases pygame super class

WIDTH, HEIGHT = 500, 500 #dimensions for menu window
FPS = 60 #number of times game loop is run per second
SQUARESIZE = WIDTH//8 #dimensions of board squares 
BLACK = (80,80,80) #rgb value for black squares
WHITE = (255,255,255) #rgb value for white squares
WIN = pygame.display.set_mode((WIDTH, HEIGHT)) #creates pygame window 
WIN.fill((0,0,0)) #fills window black
MENUFONT = pygame.font.SysFont("monaco", 32) #defines font for rendering menu text

ranksToRows = {"1": 7, "2":6, "3": 5, "4":4,  #converts chess notation
               "5":3, "6": 2, "7": 1, "8": 0} #to board indexes 
filesToCols = {"a": 0, "b": 1, "c": 2, "d": 3,
               "e": 4, "f": 5, "g": 6, "h": 7}
rowsToRanks = {val:key for key,val in ranksToRows.items()} #converts board indexes
colsToFiles = {val:key for key,val in filesToCols.items()} #to chess notation

pieceValues = {"P":1, "B":3, "N":3, "R":5, "Q":9} #used to find points gained when taking a piece (king is never 'captured') 

#Contains main game loop - should be called upon opening the game
def main(whiteRematchScore=0, blackRematchScore=0, whitePlayer=None, blackPlayer=None, aiGame=False):
    if not whitePlayer and not blackPlayer: #names will already be set if this is a rematch
        whitePlayer, blackPlayer, aiGame = Menu() #opens the main menu window, returns player names and whether it's an AI game     
    gameState = GameState() #creates GameState object 
    if aiGame:
        aiHandler = ChessAI(gameState) #instantiate ChessAI class, object will be reposnsible for making method calls to control the AI
        pygame.display.set_caption("SigmaZero - Player VS AI") #sets the name of the window
    else:
        pygame.display.set_caption("SigmaZero - Player VS Player") 
        pygame.time.set_timer(pygame.USEREVENT, 1000) #raises custom event every 1000 ms to update timers

    global WIN #ensures that changes to WIN are reflected in every function 
    global FONT1 #initialising fonts takes a relatively long time, so they are initialised here
    global FONT2 #- with a global scope since separate functions also use them
    global FONT3 
    global FONT4
    WIN = pygame.display.set_mode((800,600)) #resizes the menu window into the game window 
    #Fonts used for rendering text throughout the game
    FONT1 = pygame.font.SysFont("comicsans", 18) #move tracker font
    FONT2 = pygame.font.SysFont("monaco", 24) #player name/score font
    FONT3 = pygame.font.Font("C:\\Windows\\fonts\\Seven Segment.ttf", 24) #timer font
    FONT4 = pygame.font.Font("C:\\Windows\\fonts\\Seven Segment.ttf", 28) #rematch score font
    WIN.fill((0,0,0)) #fills the resized window black 
    
    clock = pygame.time.Clock() #clock object used for controlling refresh rate
    prefetchImages() #pre-emptively fetches and renders all the game images (stored in global dictionary)
    initSquare = None #initially no square selected
    highSquares = [] #keep track of highlighted squares
    drawBoard(initSquare, {}, highSquares) #draws the starting board and pieces
    drawPieces(gameState.board)
    legalMoves = gameState.getLegalMoves() #prefetches the legal moves for white's first move 
    moveCount = 0 #keep track of how many moves made, for use in move tracker
    whiteScore, blackScore = 0,0 #initiliase player's scores to 0
    drawScores(whiteScore, blackScore, whitePlayer, blackPlayer) #displays initial player scores and rematch scores
    drawRematchScore(whiteRematchScore, blackRematchScore) #draws rematch scores
    pygame.display.update() #displays the initialised game window 
    whiteMove = gameState.whiteMove #fetches the initial moving player from GameState object
    whiteTime, blackTime = 600,600 #initialises timers to 10 minutes
    firstMove = False #indicates if first move has been made, to know if timer should start
    run = True #condition for infinite main game loop 
    while run:
        clock.tick(FPS) #controls refresh rate
        if not aiGame: #timers aren't used in AI mode
            drawTimer(whiteTime, blackTime) #draws player timers as they update
        pygame.display.update() #draws the updated timer (no need to redraw everything, since timer only drawed over itself)
        for event in pygame.event.get(): #checks for any raised pygame events
            if event.type == pygame.QUIT: #breaks game loop if window closed 
                run = False
            if event.type == pygame.USEREVENT: #raised every second, indicates timers should be updated
                if whiteMove: #only update moving player's timer
                    if firstMove: #start timer if first move has been made
                        whiteTime -= 1 #updates white timer
                        if whiteTime == 0: #end game if timer runs out 
                            endGame(whiteMove, False, False, True, whitePlayer, blackPlayer, whiteRematchScore, blackRematchScore, aiGame)    
                            run = False
                else:
                    blackTime -= 1
                    if blackTime == 0:
                        endGame(whiteMove, False, False, True, whitePlayer, blackPlayer, whiteRematchScore, blackRematchScore, aiGame)
                        run = False
                
            if event.type == pygame.MOUSEBUTTONDOWN: #raised on mouse click
                mousePosition = pygame.mouse.get_pos() #get the (x,y) coordinates on mouse click (coordinates treat top left as origin)
                if 0<mousePosition[0]<496 and 50<mousePosition[1]<546: #validate click is within board
                    sqTemp = ((mousePosition[1]-50)//SQUARESIZE, mousePosition[0]//SQUARESIZE) #find relative position of click on board
                    pieceMoved = gameState.board[sqTemp[0]][sqTemp[1]] #find piece located at click
                    if not initSquare and pieceMoved != "": #if first square isn't set and piece moved isn't empty square
                        if (whiteMove and pieceMoved[0] == "w") or (not whiteMove and pieceMoved[0] == "b"): #if colour of piece matches moving player 
                            highSquares.clear() #clear highlighted squares from previous move
                            initSquare = sqTemp #sqTemp has been validated, so set to initial square
                            highSquares.append(initSquare) #highlight this initital square            

                    elif initSquare and sqTemp != initSquare: #only store second click if it's different to first (and first wasn't null)
                        if (initSquare, sqTemp) in legalMoves: #if tuple representing move is in list of legal moves
                            highSquares.append(sqTemp) #highlight end square 
                            if whiteMove and pieceMoved != "": #only update respective score if a piece is taken
                                whiteScore += pieceValues[pieceMoved[1]]
                            elif not whiteMove and pieceMoved != "":
                                blackScore += pieceValues[pieceMoved[1]]
                            gameState.movePiece(initSquare, sqTemp) #moves piece on board, switches moving player
                            gameState.enPassantMoves.clear() #clears the list of en passant moves since moving player changed
                            gameState.wMovesDict.clear() if whiteMove else gameState.bMovesDict.clear() #clear moving players move dict
                            legalMoves = gameState.getLegalMoves() #prefetches legal moves for new moving player
                            moveCount += 1 #move was just made so update counter
                            moveTracker(moveCount,gameState.moveTracker) #update on screen notebook of moves 
                            if gameState.Checkmate: #end game if new moving player is in checkmate
                                endGame(whiteMove, True, False, False, whitePlayer, blackPlayer, whiteRematchScore, blackRematchScore, aiGame)
                                run = False
                            elif gameState.Stalemate: #end game if game has reached stalemate
                                endGame(whiteMove, False, True, False, whitePlayer, blackPlayer, whiteRematchScore, blackRematchScore, aiGame)
                                run = False 
                            firstMove = True #indicates that white has made the first move 

                            if aiGame: #if in AI mode, automatically makes moves for black (this will be first reached after the first white move, then after the 2nd, etc) 
                                whiteMove = gameState.whiteMove #fetch the new moving player
                                #draw the updated game window (since a move was just made)
                                drawBoard(None, gameState.wMovesDict, highSquares) 
                                drawPieces(gameState.board)    
                                drawScores(whiteScore, blackScore, whitePlayer, blackPlayer) 
                                pygame.display.update()
                                highSquares.clear() #clear highlighted squares (move is being made)
                                pieceMoved, square1, square2 = aiHandler.play() #makes a move for black, returns piece moved and it's initial/end squares
                                highSquares.append(square1) #highlights the move
                                highSquares.append(square2)
                                if pieceMoved != "":
                                    blackScore += pieceValues[pieceMoved[1]] #updates the AI's score if it took a piece
                                #the same process for after a normal player makes a move is then used:
                                gameState.enPassantMoves.clear() 
                                gameState.bMovesDict.clear() #clear moving players move dict
                                legalMoves = gameState.getLegalMoves()
                                moveCount += 1
                                moveTracker(moveCount,gameState.moveTracker)
                                if gameState.Checkmate:
                                    endGame(whiteMove, True, False, False, whitePlayer, blackPlayer, whiteRematchScore, blackRematchScore, aiGame)
                                    run = False
                                elif gameState.Stalemate:
                                    endGame(whiteMove, False, True, False, whitePlayer, blackPlayer, whiteRematchScore, blackRematchScore, aiGame)
                                    run = False
                                
                        initSquare = None #resets the initial square, since either a move was made or the move was illegal 
                        whiteMove = gameState.whiteMove #fetch the new moving player
                    #redraw all the onscreen entities, even if they haven't changed (since some objects are drawn over eachother)
                    drawBoard(initSquare, gameState.wMovesDict if whiteMove else gameState.bMovesDict, highSquares) 
                    drawPieces(gameState.board)    
                    drawRematchScore(whiteRematchScore, blackRematchScore)
                    drawScores(whiteScore, blackScore, whitePlayer, blackPlayer)
                    pygame.display.update()

                    
                
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    gameState.revMove()

                        
                    
                    
                    
    pygame.quit()

#The main menu
def Menu():
    menuMode = True #indicates that the player is on the main menu (not a sub-menu)
    pygame.display.set_caption("SigmaZero") #sets name of window
    #fetches the background images for the menus  
    menuImage = pygame.image.load(os.path.abspath("./pieces/menu1.png"))
    setNamesImage = pygame.image.load(os.path.abspath("./pieces/setNames.png"))
    WIN.blit(menuImage, (0,0)) #initially displays the main menu image
    player1Active = False #indicates whether player1 input field selected (in set names menu)
    player2Active = False #indicates whether player2 input field selected (in set names menu)
    player1 = [] #tracks keys entered in player1 name field (in set names menu)
    player2 = [] #tracks keys entereed in player2 name field (in set names menu)
    run = True
    while run: #continuously displays menu until closed 
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: #if window closed
                sys.exit("Program closed by user") #since closing this window normally will just cause the program 
                                                   #to return to main(), sys.exit() is used to close the whole program 
            if event.type == pygame.MOUSEBUTTONDOWN: #raised on mouse click
                mousePosition = pygame.mouse.get_pos() #gets (x,y) coordinates of mouse click
                if 46<mousePosition[0]<203 and 312<mousePosition[1]<400 and menuMode: #relative position of PvP button
                    menuMode = False #entering submenu, so set main menu flag to False
                    pygame.display.set_caption("SigmaZero - Set names") #update name of window 
                    WIN.blit(setNamesImage, (0,0)) #display the background to the set names image
                    #draws two rectangles to act as backgrounds for the name input fields
                    player1Rect = pygame.Rect(130,215,240,30)
                    player2Rect = pygame.Rect(130,270,240,30)
                    pygame.draw.rect(WIN, (0,0,0), player1Rect)
                    pygame.draw.rect(WIN, (0,0,0), player2Rect)
                elif 286<mousePosition[0]<443 and 312<mousePosition[1]<400 and menuMode: #relative position of PvAI button
                    return "Human", "Computer", True #returns 2 default names and 'True' since it's an AI game
                if not menuMode: #in the set names sub menu
                    if player1Rect.collidepoint(mousePosition): #mouse click on player1 name input field
                        pygame.draw.rect(WIN, (0,0,0), player2Rect) #unselects player2 field (black rect)
                        WIN.blit(player2Text, (135,275)) if len(player2)>0 else None #redraws player 2 text (was drawn over by rect)
                        pygame.draw.rect(WIN, (70,70,70), player1Rect)  #highlights player1 input field (grey rect)
                        WIN.blit(player1Text, (135,220)) if len(player1)>0 else None #redraws player 1 text (was drawn over by rect)
                        player1Active = True #indicates player1 input field selected
                        player2Active = False #indicates player2 input field not selected
                
                    elif player2Rect.collidepoint(mousePosition): #mouse click on player2 name input field
                        pygame.draw.rect(WIN, (0,0,0), player1Rect)  #unselects player1 field (black rect)
                        WIN.blit(player1Text, (135,220)) if len(player1)>0 else None #redraws player 1 text (was drawn over by rect)
                        pygame.draw.rect(WIN, (70,70,70), player2Rect) #highlights player2 input field (grey rect)
                        WIN.blit(player2Text, (135,275)) if len(player2)>0 else None #redraws player 2 text (was drawn over by rect
                        player1Active = False #indicates player1 input field not selected
                        player2Active = True #indicates player2 input field now selected

                if 330<mousePosition[0]<410 and 322<mousePosition[1]<370 and not menuMode: #'Done' button pressed 
                    if 0<len(player1)<=8 and 0<len(player2)<=8 and player1!=player2: #validates names are correct length and not the same
                        return "".join(player1), "".join(player2), False #returns the combined character lists (names) and 'False' (Pvp mode) 
                    else: #name/s are invalid
                        WIN.blit(setNamesImage, (0,0)) #redraws background image
                        #unselects both input fields
                        pygame.draw.rect(WIN, (0,0,0), player1Rect)
                        pygame.draw.rect(WIN, (0,0,0), player2Rect)
                        #redraws any inputted names, or nothing at all if they don't exist (len(player1/2)==0)
                        WIN.blit(player1Text, (135, 220)) if len(player1)>0 else None
                        WIN.blit(player2Text, (135,275)) if len(player2)>0 else None
                        #displays the relevant error with the inputted names
                        if len(player1)==0 or len(player2) == 0:
                            errorText = MENUFONT.render("Please enter name for player 1/2", True, (255,0,0))
                        elif len(player1) > 8 or len(player2) > 8:
                            errorText = MENUFONT.render("Please reduce length of player name", True, (255,0,0))
                        else:
                            errorText = MENUFONT.render("Please change one of the names", True, (255,0,0))
                        WIN.blit(errorText, (100,400)) 
            if event.type == pygame.KEYDOWN and not menuMode: #keyboard pressed whilst in enter name submenu
                if player1Active: #player1 name input field selected
                    if event.key == pygame.K_BACKSPACE: #backspace key pressed
                        player1.pop() if len(player1)>0 else None #removes the last element of player1's name if it exists
                        player1Text = MENUFONT.render("".join(player1), True, WHITE) #renders updated player1 text 
                        pygame.draw.rect(WIN, (70,70,70), player1Rect) #covers up previous text
                        WIN.blit(player1Text, (135, 220)) #draws rendered text to player1 input field
                    elif event.unicode.isalpha() or event.unicode.isdigit(): #checks that inputted character is letter or number
                        player1.append(str(event.unicode)) #adds the inputted character to player1's name 
                        player1Text = MENUFONT.render("".join(player1), True, WHITE) #renders updated player1 text
                        WIN.blit(player1Text, (135, 220)) #draws rendered text to player1 input field

                elif player2Active: #player2 name input field selected
                    if event.key == pygame.K_BACKSPACE: #backspace key pressed 
                        player2.pop() if len(player2)>0 else None #removes the last element of player2's name if it exists
                        player2Text = MENUFONT.render("".join(player2), True, WHITE) #renders updated player2 text 
                        pygame.draw.rect(WIN, (70,70,70), player2Rect)
                        WIN.blit(player2Text, (135, 275)) #draws rendered text to player2 input field
                    elif event.unicode.isalpha() or event.unicode.isdigit(): #checks that inputted character is letter or number
                        player2.append(str(event.unicode)) #adds the inputted character to player2's name 
                        player2Text = MENUFONT.render("".join(player2), True, WHITE) #renders updated player2 text
                        WIN.blit(player2Text, (135, 275)) #draws rendered text to player2 input field
            
    
#Post-game screen
def endGame(whiteMove, Checkmate, Stalemate, Timeout, whitePlayer, blackPlayer, whiteScore, blackScore, aiGame):
    WIN = pygame.display.set_mode((500,500)) #resizes game window to menu size
    WIN.fill((0,0,0)) #fills the window black
    pygame.display.set_caption(("SigmaZero - Post game")) #updates window name
    colour = "w" if whiteMove else "b" #string denoting moving player (first character of post-game image's names)
    if Checkmate: #game ended by checkmate
        endImage = IMAGES[colour+"Checkmate"] #fetches respective checkmate post-game screen 
        winner = colour
    elif Stalemate: #game ended by stalemate
        winner = None #game is a draw so no winner
        endImage = IMAGES["Stalemate"] #fetches stalemate post-game screen
    elif Timeout: #game ended on time
        winner = "b" if whiteMove else "w" #opposing player wins if moving player's timer runs out
        endImage = IMAGES[colour+"Timeout"] #fetches respective timer post-game screen
    WIN.blit(endImage, (0,0)) #displays the fetched post-game screen
    run = True
    while run: #continuously displays post-game screen until closed
        pygame.display.update() #updates window with new post-game screen
        for event in pygame.event.get():
            if event.type == pygame.QUIT: #if window closed
                sys.exit("Program closed by user") #since closing this window normally will just cause the program 
                                                   #to return to main(), sys.exit() is used to close the whole program 
            if event.type == pygame.MOUSEBUTTONDOWN: #raised on mouse click
                mousePosition = pygame.mouse.get_pos() #get (x,y) coordinates of mouse click
                if 46<mousePosition[0]<203 and 312<mousePosition[1]<450: #relative position of return to menu button
                    run = False #breaks out loop, causes interpreter to reach 'main()' on last line
                elif 290<mousePosition[0]<447 and 312<mousePosition[1]<450: #relative position of rematch button
                    #calls main() again, passing the updated scores and the current player names
                    if winner == "w": #white won
                        main(int(whiteScore)+1, int(blackScore), whitePlayer, blackPlayer, aiGame)#update score of winning player and reuse names
                        return 0
                    elif winner=="b": #black won
                        main(int(whiteScore), int(blackScore)+1, whitePlayer, blackPlayer, aiGame)#update score of winning player and reuse names
                        return 0
                    else: #game was a draw 
                        main(int(whiteScore), int(blackScore), whitePlayer, blackPlayer, aiGame) #scores unchanged since it was a draw
                        return 0
                    

    main() #restarts the game, returning to the main menu, if the player clicked 'return to menu'




#On-screen notebook
def moveTracker(moveCount, moveTracker): 
    if moveTracker != []: #if a move has been made
        move = moveTracker[moveCount-1] #access most recent move
        #Converts the move into a string of chess notation and renders it into a pygame object 
        move = "{}. {}{} {}{}".format(str(moveCount),str(colsToFiles[move[0][1]]), str(rowsToRanks[move[0][0]]), str(colsToFiles[move[1][1]]), str(rowsToRanks[move[1][0]]))
        moveText = FONT1.render(move, True, WHITE) 
        #draws the rendered text to a newline on the window, or to a new column if the previous one is full 
        if moveCount%27 == 0: #handles the edge case of displaying the last line in each column (27 rows in each column)
            WIN.blit(moveText, (500 + (moveCount//27 - 1)*80,586)) #y=586 is the last row in each column, moveCount//27 is the line's column +1, so we subtract 1
        else: #.blit() draws the rendered object to the window, with it's top-left corner at the passed (x,y) coordinates
            WIN.blit(moveText, (500 + (moveCount//27)*80,(moveCount%27 -1)*20+70)) #first column has x=500 and y=70, each row is 20 pixels apart and each column is 80.
        


#Prefetches and renders game images
def prefetchImages():
    global IMAGES #the images are used throughout the game's functions 
    #loads each image using their respective path, renders them into a pygame object, and then maps them to a key in a dictionary (IMAGES)
    keys = ["bP", "wP", "bR", "wR", "bK", "wK", "bQ", "wQ", "bB", "wB", "bN", "wN", "wCheckmate", "bCheckmate", "wForfeit", "bForfeit", "Stalemate", "wTimeout", "bTimeout"]
    images = [pygame.image.load(os.path.abspath("./pieces/"+key+".png")) for key in keys]
    IMAGES = dict(zip(keys, images))
    

#draws the chess board and it's highlighted squares
def drawBoard(square, movesDict, highSquares=[]):
    if square: #if initial square has just been selected
        allMoves = movesDict[str(square)] #accesses moves with 'square' as an origin
    else:
        allMoves = [] #move has already been made or no squares have been selected yet
    #iterate through board's rows and columns
    for row in range(1,9):
        for col in range(1,9):
            rect = pygame.Rect((row-1)*SQUARESIZE, (col-1)*SQUARESIZE+50, SQUARESIZE, SQUARESIZE) #pygame rectangle (square) object to be drawn
            if (row+col)%2 == 0: #white square
                if (col-1, row-1) in highSquares: #if square is in list of highlighted squares
                    pygame.draw.rect(WIN, (255,232,124), rect) #highlights square in lighter shade of yellow to if square is black
                elif (col-1, row-1) in allMoves: #if square is an endsquare of one of the moves
                    pygame.draw.rect(WIN, (65, 105, 225), rect) #highlights square in lighter shade of blue to if square is black
                else: #if square isn't to be highlighted 
                    pygame.draw.rect(WIN, WHITE, rect) #draws white square to the window 
            else: #black square
                if (col-1, row-1) in highSquares:
                    pygame.draw.rect(WIN, (255,216,1), rect)
                elif (col-1, row-1) in allMoves:
                    pygame.draw.rect(WIN, (46, 76, 165), rect)
                else:
                    pygame.draw.rect(WIN, BLACK, rect)
                
def drawPieces(board):
    #iterates through the pieces on the board 
    for row in range(8):
        for col in range(8):
            piece = board[row][col] #accesses square at current (row,col)
            if piece != "": #if there is a piece on the current square
                image = IMAGES[piece] #fetches the rendered image for the piece
                WIN.blit(image, ((col)*SQUARESIZE, (row)* SQUARESIZE+50)) #draws the image to the board

#Draws the player's names and scores 
def drawScores(whiteScore, blackScore, player1, player2):
    #background rectangles for the names and scores to be drawn on top of
    rect1 = pygame.Rect(0, 0, 496, 50) 
    rect2 = pygame.Rect(0,546,496,54)
    #renders the names and scores text (with aliasing enabled), white text in red, black text in green
    player1Text = FONT2.render(f"|White: {player1}|", True, (255,0,0))
    player2Text = FONT2.render(f"|Black: {player2}|", True, (0,255,0))
    score1Text = FONT2.render(f"|Score: {whiteScore}|", True, (255,0,0))
    score2Text = FONT2.render(f"|Score: {blackScore}|", True, (0,255,0))
    #draws the background rectangles and rendered text 
    pygame.draw.rect(WIN, (95,95,95), rect1)
    pygame.draw.rect(WIN, (235,235,235), rect2)
    WIN.blit(player2Text, (10,20))
    WIN.blit(player1Text, (10,570))
    WIN.blit(score2Text, (200, 20))
    WIN.blit(score1Text, (200, 570))

#Draws the time left for each player
def drawTimer(whiteTime, blackTime):
    #Renders the time left text for each player in minutes:seconds 
    whiteTime = FONT3.render(f"{whiteTime//600}{(whiteTime%600)//60}:{(whiteTime%60)//10}{(whiteTime%60)%10}", True, (255,0,0))   
    blackTime = FONT3.render(f"{blackTime//600}{(blackTime%600)//60}:{(blackTime%60)//10}{(blackTime%60)%10}", True, (0,255,0)) 
    #background rectangles for the timers to be drawn on top of
    whiteRect = pygame.Rect(350,563,120,30)
    blackRect = pygame.Rect(350,13,120,30)
    #draws the background rectangles and timer text to the window, white text in red, black text in green
    pygame.draw.rect(WIN, (205,205,205), whiteRect)
    pygame.draw.rect(WIN, (65,65,65), blackRect)
    WIN.blit(whiteTime, (380, 565))
    WIN.blit(blackTime, (380, 15))

#Draws the rematch scores for the players 
def drawRematchScore(whiteScore, blackScore):
    #Renders the score text for each player (separately since in different colours) 
    whiteText = FONT4.render(str(whiteScore), True, (255,0,0)) #white text in red
    blackText = FONT4.render(str(blackScore), True, (0,255,0)) #black text in green
    #Renders additonal text for readability (a separator between the scores and subheading) 
    separatorText = FONT4.render(":", True, (0,0,0))
    rematchText = FONT1.render("Rematch score:", True, (0,0,0))
    #draws a background rectangle for the text to be drawn on top of 
    rect = pygame.Rect(496,0,304,70)
    pygame.draw.rect(WIN, (100,100,100), rect)
    #draws the rendered text to the window (on top of the background rectangle)
    WIN.blit(rematchText, (500,5))
    WIN.blit(whiteText, (600, 20))
    WIN.blit(separatorText, (630, 20))
    WIN.blit(blackText, (650, 20))
    

    


#Implements all functionality for the the chess game itself
class GameState:

    def __init__(self):
        #initialises all pieces to their starting positions
        self.board = np.array([["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
                               ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"], 
                               ["", "", "", "", "", "", "", ""],
                               ["", "", "", "", "", "", "", ""], 
                               ["", "", "", "", "", "", "", ""], 
                               ["", "", "", "", "", "", "", ""],
                               ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
                               ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]])
        self.moveFuncs = {"R":"self.getRookMoves", "N":"self.getKnightMoves", "B":"self.getBishopMoves", 
                          "Q":"self.getQueenMoves", "K":"self.getKingMoves", "P":"self.getPawnMoves"}  #used to quickly map squares to move functions
        
        self.wKPosition = (7,4) #tracks white king position
        self.bKPosition = (0,4) #tracks black king position
        self.whiteMove = True #current moving player (True if white)
        self.moveTracker = [] #list of moves made so far
        self.Checkmate = False #whether the game has reached checkmate (True if so)
        self.Stalemate = False  #whether the game has reached stalemate (True if so)
        self.bkCastling = True #black king side castling possible
        self.bqCastling = True #black queen side castling possible
        self.wkCastling = True #white king side castling possible
        self.wqCastling = True #white queen side castling possible
        self.doubleMove = False #indicates whether last move was a pawn double move
        self.enPassantMoves = [] #list of possible en passant moves the moving player could make 
        self.wMovesDict = {} #maps white pieces to their possible moves, used in highlighting possible moves (in drawBoard())
        self.bMovesDict = {} #maps black pieces to their possible moves, used in highlighting possible moves (in drawBoard())
        self.moveCache = {} #maps move to opponent's moves found when inCheck() called (to be reused on opponent's move)

    #Handles updating the board as moves are made
    def movePiece(self, initSquare, endSquare): #takes move's initial and end square
        track = True #whether the move should be appended to moveTracker at the end (set to False for castling)
        self.currentDoubleMove = bool(self.doubleMove) #caches the current state of doubleMove in case move is reversed
        self.doubleMove = False #since this a new move, resets flag (will be set to True if move is found to be doubleMove)
        enPasMove = False #used in revMove() to when to reverse en passant moves (set to True if it is)
        wCurrentCastling = (self.wkCastling, self.wqCastling) #caches the current state of wcastling flags in case move is reversed
        bCurrentCastling = (self.bkCastling, self.bqCastling) #caches the current state of bcastling flags in case move is reversed
        pieceMoved = self.board[initSquare[0]][initSquare[1]] #accesses piece which is being moved
        endPiece = self.board[endSquare[0]][endSquare[1]] #accesses piece at square which piece is moving to

        #Handles pawn promotion (automatic queen)
        if pieceMoved == "wP" and endSquare[0] == 0:
            self.board[initSquare[0]][initSquare[1]], self.board[endSquare[0]][endSquare[1]] = "", "wQ"
        elif pieceMoved == "bP" and endSquare[0] == 7:
            self.board[initSquare[0]][initSquare[1]], self.board[endSquare[0]][endSquare[1]] = "", "bQ"

        else:
            #Handles castling, each branch performs the same function (just on different squares)
            if pieceMoved == "wK":
                if initSquare == (7,4) and endSquare == (7,2): #castling move
                    self.board[7][0], self.board[7][3] = "", "wR" #updates rook postion as well 
                    #tracks the move for both the king and rook, boolean parameter (5th) indicates this is a castling move 
                    self.moveTracker += (((7,0), (7,3), "wR", "", False, wCurrentCastling, bCurrentCastling),(initSquare,endSquare,pieceMoved,endPiece, True, wCurrentCastling, bCurrentCastling, enPasMove))
                    track = False #since move already added to moveTracker (otherwise will be tracked twice)
                elif initSquare == (7,4) and endSquare == (7,6):
                    self.board[7][7], self.board[7][5] = "", "wR"
                    self.moveTracker += (((7,7), (7,5), "wR", "", False, wCurrentCastling, bCurrentCastling),(initSquare,endSquare,pieceMoved,endPiece, True, wCurrentCastling, bCurrentCastling, enPasMove))
                    track = False
                self.wKPosition = endSquare #updates the king position 
                self.wkCastling, self.wqCastling = False, False #king and rook both moved, so castling no longer possible
            elif pieceMoved == "bK":
                if initSquare == (0,4) and endSquare == (0,2):
                    self.board[0][0], self.board[0][3] = "", "bR"
                    self.moveTracker += (((0,0), (0,3), "bR", "", False, wCurrentCastling, bCurrentCastling),(initSquare,endSquare,pieceMoved,endPiece, True, wCurrentCastling, bCurrentCastling, enPasMove))
                    track = False
                elif initSquare == (0,4) and endSquare == (0,6):
                    self.board[0][7], self.board[0][5] = "", "bR"
                    self.moveTracker += (((0,7), (0,5), "bR", "", False, wCurrentCastling, bCurrentCastling),(initSquare,endSquare,pieceMoved,endPiece, True, wCurrentCastling, bCurrentCastling, enPasMove))
                    track = False
                self.bKPosition = endSquare 
                self.bkCastling, self.bqCastling = False, False

            #disables respective side's castling flag when white rook moved
            elif pieceMoved == "wR":
                if initSquare == (7,0): #queenside rook
                    self.wqCastling = False
                elif initSquare == (7,7): #kingside rook
                    self.wkCastling = False

            #disables respective side's castling flag when black rook moved
            elif pieceMoved == "bR":
                if initSquare == (0,0): #queenside rook
                    self.bqCastling = False
                elif initSquare == (0,7): #kingside rook
                    self.bkCastling = False
            
            elif pieceMoved[1] == "P" and abs(initSquare[0] - endSquare[0]) == 2: #pawn double move
                self.doubleMove = True #indicates this move was a double pawn advance
            
            elif (initSquare, endSquare) in self.enPassantMoves: #move is an en passant move
                self.board[initSquare[0]][endSquare[1]] = "" #removes the taken piece from board
                enPasMove = True #indicates to revMove() that it needs to reverse an en passant move
            
            self.board[initSquare[0]][initSquare[1]], self.board[endSquare[0]][endSquare[1]] = "", pieceMoved #reflects the move on the board (moves the piece)
        if track: #only tracks moves if they havent already (if it was castling)
            self.moveTracker.append((initSquare,endSquare,pieceMoved,endPiece, False, wCurrentCastling, bCurrentCastling, enPasMove)) #passes all the information needed to display or reverse moves
        self.whiteMove = not self.whiteMove #move has been made so switch moving player
        

    #Used to reverse moves when checking legality
    def revMove(self):
        previousMove = self.moveTracker.pop() #most recent move stored on top of move tracker list
        if previousMove[4]: #denotes whether castling move, if it is pop the next move also (the rook move)  
            rookMove = self.moveTracker.pop()
            self.board[rookMove[0][0]][rookMove[0][1]], self.board[rookMove[1][0]][rookMove[1][1]] = rookMove[2], "" #reverses the rook move(on board)
        #reverses the move's effect on the board
        self.board[previousMove[0][0]][previousMove[0][1]], self.board[previousMove[1][0]][previousMove[1][1]] = previousMove[2], previousMove[3]  
        if previousMove[2][0] == "w": #white piece moved
            self.wkCastling, self.wqCastling = previousMove[5] #reverts wcastling flags to cached ones (from before move was made)
            if previousMove[2] == "wK": #piece moved was white king
                self.wKPosition = previousMove[0] #reverts wking position tracker
            elif previousMove[7]: #en passant move
                self.board[previousMove[0][0]][previousMove[1][1]] = "bP" #re-places the taken black pawn 
        else: #black piece moved
            self.bkCastling, self.bqCastling = previousMove[6] #reverts bcastling flags to cached ones (from before move was made)
            if previousMove[2] == "bK": #piece moved was black king
                self.bKPosition = previousMove[0] #reverts bking postion tracker
            elif previousMove[7]: #en passant move
                self.board[previousMove[0][0]][previousMove[1][1]] = "wP" #re-places the taken white pawn
        self.doubleMove = bool(self.currentDoubleMove) #reverts doubleMove flag to cache one (from before move was made)
        self.whiteMove = not self.whiteMove #move reversed so reverse moving player

    #Parses all the moves moving player could make without check limitations
    def getAllMoves(self):
        moves = [] #the possible moves
        #iterates through each square on the board 
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col] 
                if piece != "": #if square isn't empty (piece is present)
                    colour = piece[0] #first character of piece is 'w' or 'b' (colour)
                    if (colour == "w" and self.whiteMove) or (colour == "b" and not self.whiteMove): #only resolve moves for pieces belonging to moving player
                        eval(self.moveFuncs[piece[1]]+"(row,col,colour,moves)") #accesses name of relevant move function (and calls it - updating the move list)  
        return moves #returns the list of parsed moves 
    
    #Updates the list of possible moves by taking legality into account
    def getLegalMoves(self, aiMove=False): #aiMove set to True when called by ChessAI.minMax()
        self.currentPlayer = bool(self.whiteMove) #accesses value of moving player (prevents modifying object reference)
        if aiMove: #moveCache is not used (defined) when parsing AI moves, so possible moves fetched directly 
            allMoves = self.getAllMoves() 
        else: #fetching moves for human player
            #accesses the cached copy of possible moves stemming from the last move, or directly calls getAllMoves() if a move has yet to be made 
            allMoves = [i for i in self.moveCache[str((self.moveTracker[-1][0], self.moveTracker[-1][1]))]] if self.moveCache else self.getAllMoves()
            self.moveCache.clear() #possible moves now been accessed, so cache cleared (ready to receive the new possible moves)
        for moveIndex in range(len(allMoves)-1, -1, -1): #iterates backwards through move list since elements are being removed
            move = allMoves[moveIndex] #accesses current possible move from move list
            self.movePiece(move[0], move[1]) #makes the currently accessed move 
            self.whiteMove = not self.whiteMove #movePiece inverted moving players, need to revert back
            inCheck, oppMoves = self.kingInCheck() #returns whether the king is in check after the move, and the list of opponent's possible moves (after move was made)
            if inCheck == True: #king is in check after move was made
                allMoves.remove(move) #if king still in check after move, move is illegal so remove it
                if self.currentPlayer: #white move
                    self.wMovesDict[str(move[0])].remove(move[1]) if move[1] in self.wMovesDict[str(move[0])] else None#update wmoveDict aswell 
                else: #black move
                    self.bMovesDict[str(move[0])].remove(move[1]) if move[1] in self.bMovesDict[str(move[0])] else None#update bmoveDict aswell 
            else: #move was legal
                self.moveCache[str(move)] = oppMoves #caches the possible opponent moves (only if the original move was legal)
            self.revMove() #reverse the effects of the move made
            self.whiteMove = not self.whiteMove #revMove() switches moving players, need to revert back
        if len(allMoves) == 0: #there are no legal moves that can be made
            if inCheck == True: 
                self.Checkmate = True #since no legal moves and player is in check, it's checkmate
            else:
                self.Stalemate = True #since no legal moves but the player isn't in chekc, it's a stalemate
        return allMoves #returns the list of legal moves

    #Determines if the given square is under attack
    def underAttack(self, square, kingCheck = False): #kingCheck passed as True when validating if king is in check
        self.whiteMove = not self.whiteMove #switches to opponent
        moves = [i for i in self.getAllMoves()] #gets opponent's possible moves in current position
        self.whiteMove = not self.whiteMove #reverts back to current moving player
        for move in moves: #iterates through opponent's possible moves 
            if move[1] == square: #endSquare of opponent's move lands on the given square (under attack)
                return True, moves if kingCheck else True #returns that the square is under attack (and the fetched opponent moves if called to detect check)
        return False, moves if kingCheck else False #none of the opponent's moves landed on the square, so it's not under attack

    #Determines if the moving player's king is in check
    def kingInCheck(self): 
        if self.whiteMove: #white is moving
            return self.underAttack(self.wKPosition, True) #returns whether current wking possion is under attack 
        else: #black is moving
            return self.underAttack(self.bKPosition, True) #returns whther current bking position is under attack

    #Determines if castling is possible for moving player
    def castlingCheck(self, moves, pieceMoves): 
        #each branch performs the same function, just for different sides and colours
        if self.whiteMove: #white is moving
            if self.wqCastling: #wqueen side castling is possible
                if self.board[7][1] == "" and self.board[7][2] == "" and self.board[7][3] == "": #check that no pieces are blocking path
                    if True not in list(map(self.underAttack, [(7, i) for i in range(0,5)])): #check that none of the squares between are under attack (and current king square)
                        move = ((7,4),(7,2)) #the king move
                        moves.append(move) #add move for king
                        pieceMoves.append(move[1]) #end square of king move
                        
            if self.wkCastling: #wking side castling is possible
                if self.board[7][6] == "" and self.board[7][5] == "": #check that no pieces are blocking path
                    if True not in list(map(self.underAttack, [(7, i) for i in range(4,8)])): #check that none of the squares between are under attack (and current king square)
                        move = ((7,4), (7,6)) #the king move
                        moves.append(move) #add move for king
                        pieceMoves.append(move[1]) #end square of king move
        else: #black is moving
            if self.bqCastling: #bqueen side castling
                if self.board[0][1] == "" and self.board[0][2] == "" and self.board[0][3] == "": #check that no pieces are blocking path
                    if True not in list(map(self.underAttack, [(0, i) for i in range(0,5)])): #check that none of the squares between are under attack (and current king square)
                        move = ((0,4), (0,2)) #theking move
                        moves.append(move) #add move for king
                        pieceMoves.append(move[1]) #end square of king move
            if self.bkCastling: #bking side castling
                if self.board[0][6] == "" and self.board[0][5] == "": #check that no pieces are blocking path
                    if True not in list(map(self.underAttack, [(0, i) for i in range(4,8)])): #check that none of the squares between are under attack (and current king square)
                        move = ((0,4), (0,6)) #the king move
                        moves.append(move) #add moves for king
                        pieceMoves.append(move[1]) #end square of king move
        
    #Determines the possible moves for a given rook
    def getRookMoves(self, row, col, colour, moves, queenCall=False, pieceMoves = []):
        #Each for loop performs the same operation, just in different directions

        for i in range(1, 8-row): #iterates through rows downwards
            piece = self.board[row+(i*1)][col] #contents of current end square
            move = ((row, col),(row+(i*1), col)) #the move (initSquare, endSquare)
            if piece == "": #square is empty
                moves.append(move) #move is possible, so add it to move list
                pieceMoves.append(move[1]) #add end square to list of current piece moves
            elif piece[0] != colour: #opponent's piece
                moves.append(move) #move is possible, so add it to move list
                pieceMoves.append(move[1]) #add end square to list of current piece moves
                break #rook can't jump over piece, so stop searching
            else: #if the piece is the same colour as moving player, stop search
                break
             
        for j in range(1, 8-col): #iterates through columns to the right
            piece = self.board[row][col+(j*1)]
            move = ((row, col),(row, col+(j*1)))
            if piece == "":
                moves.append(move)
                pieceMoves.append(move[1])
            elif piece[0] != colour:
                moves.append(move)
                pieceMoves.append(move[1])
                break
            else:
                break
                
        for z in range(1, 9-(8-row)): #iterates through rows upwards
            piece = self.board[row-(1*z)][col]
            move = ((row, col),(row-(z*1), col))
            if piece == "":
                moves.append(move)
                pieceMoves.append(move[1])
            elif piece[0] != colour:
                moves.append(move)
                pieceMoves.append(move[1])
                break
            else:
                break

        for w in range(1, 9-(8-col)): #iterates through columns to the left
            piece = self.board[row][col-(1*w)]
            move = ((row, col),(row, col-(w*1)))
            if piece == "":
                moves.append(move)
                pieceMoves.append(move[1])
            elif piece[0] != colour:
                moves.append(move)
                pieceMoves.append(move[1])
                break
            else:
                break
        
        #Stores all the end squares of this piece's possible moves, for use in move highlighting 
        if not queenCall: #function wasn't being used to parse queen moves (by getQueenMoves())
            if self.currentPlayer == self.whiteMove: #function isn't being called via underAttack() (moving player hasnt been switched)
                if self.whiteMove: #white to move
                    self.wMovesDict[str((row,col))] = [i for i in pieceMoves] #maps the end squares of the possible moves to this piece 
                else: #black to move
                    self.bMovesDict[str((row,col))] = [i for i in pieceMoves] #maps the end squares of the possible moves to this piece 
            else: #function has been been called via underAttack() (moving player has been switched)
                if self.currentPlayer:
                    self.bMovesDict[str((row,col))] = [i for i in pieceMoves] #maps the end squares of the possible moves to this piece 
                else:
                    self.wMovesDict[str((row,col))] = [i for i in pieceMoves] #maps the end squares of the possible moves to this piece 
            pieceMoves.clear() #ensures squares aren't highlighted when opponent's rook selected later

                
    #Determines the possible moves for a given knight
    def getKnightMoves(self, row, col, colour, moves):
        pieceMoves = []  #keep track of possible moves for this specific piece, to be added to moveDict
        moveVectors = ((1,2),(-1,2),(1,-2),(-1,-2),(2,1),(-2,1),(2,-1),(-2,-1)) #the different ways the knight can move
        for v in moveVectors: #iterates through possible move vectors
            endRow = row + v[0]
            endCol = col +v[1]
            move = ((row, col), (endRow, endCol)) #the proposed move
            if 0<=endRow<8 and 0<=endCol<8: #end square is on the board
                piece = self.board[endRow][endCol] #contents of the end square 
                if piece == "": #end square is empty
                    moves.append(move) #move is possible so add it to list
                    pieceMoves.append(move[1]) #add end square to list of current piece moves
                elif piece[0] != colour: #lands on opponent's piece
                    moves.append(move) #move is legal so add it to list
                    pieceMoves.append(move[1]) #add end square to list of current piece moves
        #Stores all the end squares of this piece's possible moves, for use in move highlighting 
        if self.currentPlayer == self.whiteMove: #function isn't being called via underAttack() (moving player hasnt been switched)
            if self.whiteMove:
                self.wMovesDict[str((row,col))] = [i for i in pieceMoves] #maps the end squares of the possible moves to this piece
            else:
                self.bMovesDict[str((row,col))] = [i for i in pieceMoves] #maps the end squares of the possible moves to this piece
        else: #function has been been called via underAttack() (moving player has been switched)
            if self.currentPlayer:
                self.bMovesDict[str((row,col))] = [i for i in pieceMoves] #maps the end squares of the possible moves to this piece
            else:
                self.wMovesDict[str((row,col))] = [i for i in pieceMoves] #maps the end squares of the possible moves to this piece
                
                
                
    #Determines the possible moves for a given bishop
    def getBishopMoves(self, row, col,colour, moves, queenCall=False, pieceMoves = []):
        moveVectors = ((1,1),(1,-1),(-1,1),(-1,-1)) #different ways the bishop can move
        for v in moveVectors: #iterates through possible move vectors
            for i in range(1,8): #bishop can move by multiples of the move vectors at once
                endRow = row + (v[0]*i)
                endCol = col + (v[1]*i)
                move = ((row, col), (endRow, endCol)) #the proposed move
                if 0<=endRow<8 and 0<=endCol<8: #end square is on the board
                    piece = self.board[endRow][endCol] #contents of the board at end square
                    if piece == "": #end square is empty
                        moves.append(move) #move is legal so add it to move list
                        pieceMoves.append(move[1]) #add end square to list of current piece moves
                    elif piece[0] != colour: #lands on opponent piece
                        moves.append(move) #move is legal so add it to move list
                        pieceMoves.append(move[1]) #add end square to list of current piece moves
                        break #bishop can't jump over pieces, so stop searching
                    else: #is friendly piece 
                        break #bishop can't jump over pieces, so stop searching
        #Stores all the end squares of this piece's possible moves, for use in move highlighting 
        if not queenCall: #function wasn't being used to parse queen moves (by getQueenMoves())
            if self.currentPlayer == self.whiteMove: #function isn't being called via underAttack() (moving player hasnt been switched)
                if self.whiteMove:
                    self.wMovesDict[str((row,col))] = [i for i in pieceMoves] #maps the end squares of the possible moves to this piece
                else:
                    self.bMovesDict[str((row,col))] = [i for i in pieceMoves] #maps the end squares of the possible moves to this piece
            else: #function has been been called via underAttack() (moving player has been switched)
                if self.currentPlayer:
                    self.bMovesDict[str((row,col))] = [i for i in pieceMoves] #maps the end squares of the possible moves to this piece
                else:
                    self.wMovesDict[str((row,col))] = [i for i in pieceMoves] #maps the end squares of the possible moves to this piece
            pieceMoves.clear()

    #Determines the possible moves a given queen can make
    def getQueenMoves(self, row, col,colour, moves): 
        pieceMoves = [] #keep track of possible moves for this specific piece, to be added to moveDict
        #combinea rook and bishop functions for queen movement
        self.getRookMoves(row, col, colour, moves, True, pieceMoves)
        self.getBishopMoves(row, col, colour, moves, True, pieceMoves)
        #Stores all the end squares of this piece's possible moves, for use in move highlighting 
        if self.currentPlayer == self.whiteMove: 
            if self.whiteMove:
                self.wMovesDict[str((row,col))] = [i for i in pieceMoves]
            else:
                self.bMovesDict[str((row,col))] = [i for i in pieceMoves]
        else:
            if self.currentPlayer:
                self.bMovesDict[str((row,col))] = [i for i in pieceMoves]
            else:
                self.wMovesDict[str((row,col))] = [i for i in pieceMoves]

    #Determines the possible moves a given king can make
    def getKingMoves(self, row, col, colour, moves):
        pieceMoves = [] #keep track of possible moves for this specific piece, to be added to moveDict
        moveVectors = ((1,0),(0,1),(1,1),(-1,0),(0,-1),(-1,-1),(1,-1),(-1,1)) #different ways the king can move
        for v in moveVectors: #iterates through possible move vectors
            endRow = row + v[0]
            endCol = col + v[1]
            move = ((row, col),(endRow, endCol)) #the proposed move
            if 0<=endRow<8 and 0<=endCol<8: #end square is on the board
                piece = self.board[endRow][endCol] #contents of end square
                if piece == "": #end square is empty
                    moves.append(move) #move is legal so add it to move list
                    pieceMoves.append(move[1]) #add end square to list of current piece moves
                elif piece[0] != colour: #lands on opponent's piece
                    moves.append(move) #move is legal so add it to move list
                    pieceMoves.append(move[1]) #add end square to list of current piece moves
        self.castlingCheck(moves, pieceMoves) #checks if castling possible, updates move lists if so
        #Stores all the end squares of this piece's possible moves, for use in move highlighting 
        if self.currentPlayer == self.whiteMove: 
            if self.whiteMove:
                self.wMovesDict[str((row,col))] = [i for i in pieceMoves]
            else:
                self.bMovesDict[str((row,col))] = [i for i in pieceMoves]
        else:
            if self.currentPlayer:
                self.bMovesDict[str((row,col))] = [i for i in pieceMoves]
            else:
                self.wMovesDict[str((row,col))] = [i for i in pieceMoves]
                

    #Determines the possible moves for a given pawn
    def getPawnMoves(self, row, col,colour, moves):
        pieceMoves = [] #keep track of possible moves for this specific piece, to be added to moveDict
        #Both main branches perfrom the same function, just on different colours
        if colour == "w": #white pawn
            if row == 6 and self.board[row-2][col] == "" and self.board[row-1][col] == "": #double advance
                move = ((row, col), (row-2, col)) #the move
                moves.append(move) #double advance is possible, so add to move list
                pieceMoves.append(move[1]) #add end square to list of current piece moves
            if row == 3 and self.doubleMove: #pawn is on row which bpawns double move to, and last move was a double move
                oppPawnFile = self.moveTracker[-1][1][1] #the column which the double moving bpawn landed on
                if abs(oppPawnFile - col) == 1: #pawns are next to eachother (same row and 1 col apart)
                    enPassantMove = ((row, col), (row-1, oppPawnFile)) #the en passant move
                    moves.append(enPassantMove) #en passant is possible, so add it to move list
                    self.enPassantMoves.append(enPassantMove) #update list of possible en passant moves
                    pieceMoves.append(enPassantMove[1]) #add end square to list of current piece moves
            if row > 0: #pawn hasn't reached top of board
                if self.board[row-1][col] == "": #square above is empty
                    move = ((row, col), (row-1, col)) #the move
                    moves.append(move) #move is legal, so add it to move list
                    pieceMoves.append(move[1]) #add end square to list of current piece moves
                if col < 7: #pawn hasn't reached right edge of board
                    if self.board[row-1][col+1] != "": #square above to the right is enemy piece
                        if self.board[row-1][col+1][0] != colour:
                            move = ((row, col), (row-1, col+1)) #the move
                            moves.append(move) #move is legal so add it to move list
                            pieceMoves.append(move[1]) #add end square to list of current piece moves
                if col > 0: #pawn hasnt reached left edege of board
                    if self.board[row-1][col-1] != "": #square above to the left is enemy piece
                        if self.board[row-1][col-1][0] != colour:
                            move = ((row, col), (row-1, col-1)) #the move 
                            moves.append(move) #move is legal so add to move list
                            pieceMoves.append(move[1]) #add end square to list of current piece moves
        else: #black pawn (same function as above just on different squares)
            if row == 1 and self.board[row+2][col] == "" and self.board[row+1][col] == "":
                move = ((row, col), (row+2, col))
                moves.append(move)
                pieceMoves.append(move[1])
            if row == 4 and self.doubleMove:
                oppPawnFile = self.moveTracker[-1][1][1]
                if abs(oppPawnFile - col) == 1: #pawns are next to eachother (same row)
                    enPassantMove = ((row, col), (row+1, oppPawnFile))
                    moves.append(enPassantMove)
                    self.enPassantMoves.append(enPassantMove)
                    pieceMoves.append(enPassantMove[1])
            if row < 7:
                if self.board[row+1][col] == "":
                    move = ((row, col), (row+1, col))
                    moves.append(move)
                    pieceMoves.append(move[1])
                if col < 7:
                    if self.board[row+1][col+1] != "":
                        if self.board[row+1][col+1][0] != colour:
                            move = ((row, col), (row+1, col+1))
                            moves.append(move)
                            pieceMoves.append(move[1])
                if col > 0:
                    if self.board[row+1][col-1] != "":
                        if self.board[row+1][col-1][0] != colour:
                            move = ((row, col), (row+1, col-1))
                            moves.append(move)
                            pieceMoves.append(move[1])
        #Stores all the end squares of this piece's possible moves, for use in move highlighting 
        if self.currentPlayer == self.whiteMove: 
            if self.whiteMove:
                self.wMovesDict[str((row,col))] = [i for i in pieceMoves]
            else:
                self.bMovesDict[str((row,col))] = [i for i in pieceMoves]
        else:
            if self.currentPlayer:
                self.bMovesDict[str((row,col))] = [i for i in pieceMoves]
            else:
                self.wMovesDict[str((row,col))] = [i for i in pieceMoves]

#Handles making the AI's moves
class ChessAI():
    def __init__(self, gameState):
        self.staticEval = MyModel() #instantiates the neural network
        self.gameState = gameState #GameState object
        self.loadTrainedModel() #updates the weights/biases of the network with the optimized parameters

    #Handles making the AI's decided move  
    def play(self):
        move = self.minMax(self.gameState.board, initialCall=True) #uses minMax to decide on the move
        endPiece = self.gameState.board[move[1][0]][move[1][1]] #piece present on the end square, used to update scores
        self.gameState.movePiece(move[0], move[1]) #makes the move 
        self.gameState.whiteMove = True #switches the moving player (AI is always black)
        return endPiece, move[0], move[1] #returns piece taken (used for score updates), and the start and end square (for highlighting)
        
    #Estimates the optimum move to make from the given position
    def minMax(self, position, depth=2, maximizingPlayer=False, initialCall=False, alpha=-np.inf, beta=np.inf):
        evaluations = {} #maps each possible move (at the top of the tree) to it's estimated value
        if depth == 0 or (self.gameState.Checkmate or self.gameState.Stalemate): #leaf node has been reached, or end of game in current position
            return self.staticEval.call(self.encodeBoard(self.gameState.board)) #feeds the passed board into the trained neural network, outputs a value representing it's 'value'
        elif maximizingPlayer: #white move
            maxEval = -np.inf #initialises current highest evaluation to negative infinity
            moves = self.gameState.getLegalMoves(aiMove=True) #gets the legal moves which could be made from the current position
            random.shuffle(moves) #if moves resolve to the same evaluation, ensures the same one isn't made every time (shuffles the list)
            for child in moves: #iterates through each move which could be made
                self.gameState.movePiece(child[0], child[1]) #makes the move
                eval = self.minMax(child, depth-1, False, alpha, beta) #evaluates the board in the new position
                print(eval)
                self.gameState.revMove() #reverts back to the previous board
                maxEval = max(maxEval, eval) #tracks the highest evaluation in current branch
                alpha = max(alpha, eval) #tracks highest evaluation in entire tree
                if beta <= alpha: #alpha beta pruning
                    break #prunes the rest of the branch (stops searching it)
            return maxEval #return the maximum evaluation in current branch
        else: #black move
            minEval = np.inf #initialises current lowest evaluation to positive infinty
            moves = self.gameState.getLegalMoves(aiMove=True) #gets the legal moves which could be made from the current position
            random.shuffle(moves) #if moves resolve to the same evaluation, ensures the same one isn't made every time (shuffles the list)
            for child in moves: #iterates through each move which could be made
                self.gameState.movePiece(child[0], child[1]) #makes the move
                eval = self.minMax(child, depth-1, True, alpha, beta) #evaluates board in new position
                print(eval)
                if initialCall: #top of tree (possible moves in current position)
                    evaluations[eval.ref()] = child #keeps log of the possible moves' evaluations 
                self.gameState.revMove() #reverts back to the previous board
                minEval = min(minEval, eval) #tracks the move with the lowest evaluation in current branch
                beta = min(beta, eval) #tracks lowest evaluation in entire tree 
                if beta <= alpha: #alpha beta pruning
                    break #prunes the rest of the branch (stops searching it)
            return evaluations[minEval.ref()] if initialCall else minEval #returns the optimum move if at top of tree, otherwise just the minimum evaluation

    #Turns the board into a ndim-1 array (column vector) of integers
    def encodeBoard(self, board):
        #maps each piece to it's type
        pieceMap = {"bP":1., "bN":2., "bB":3., "bR":4., "bQ":5., "bK":6., "wP":8., "wN":9., "wB":10., "wR":11., "wQ":12., "wK":13., "":0.}
        boardArray = np.zeros(64, dtype=np.int8) #initialises the board array to just zeros
        #iterates through each piece on the board
        for row in range(8):
            for col in range(8):
                boardArray[(7-row)*8+col] = pieceMap[board[row][col]] #updates the array with the converted piece
        return boardArray #returns the encoded board 
    
    #Loads the trained model 
    def loadTrainedModel(self):
        self.staticEval.call(np.zeros(64, dtype=np.int8)) #have to instantiate weights by making initial call (just on array of zeros)
        with open("model1.txt", "r") as trainedModel: #opens the save of the trained model (each variables is on a different line)
            #loads dot layer weights and biases
            dotWs = np.array(trainedModel.readline().rstrip()[1:-1].split(","), dtype="f")
            dotBs = np.array(trainedModel.readline().rstrip()[1:-1].split(","), dtype="f")
            self.staticEval.dotWs = dotWs
            self.staticEval.dotBs = dotBs
            #loads hidden layer 1 weights and biases
            dense1Ws = np.array(trainedModel.readline().rstrip()[1:-1].split(","), dtype="f").reshape(1,2048)
            dense1Bs = np.array(trainedModel.readline().rstrip()[1:-1].split(","), dtype="f")
            self.staticEval.dense1.set_weights([dense1Ws, dense1Bs])
            #loads hidden layer 2 weights and biases
            dense2Ws = np.array(trainedModel.readline().rstrip()[1:-1].split(","), dtype="f").reshape(2048,2048)
            dense2Bs = np.array(trainedModel.readline().rstrip()[1:-1].split(","), dtype="f")
            self.staticEval.dense2.set_weights([dense2Ws, dense2Bs])
            #loads hidden layer 3 weights and biases 
            dense3Ws = np.array(trainedModel.readline().rstrip()[1:-1].split(","), dtype="f").reshape(2048,2048)
            dense3Bs = np.array(trainedModel.readline().rstrip()[1:-1].split(","), dtype="f")
            self.staticEval.dense3.set_weights([dense3Ws, dense3Bs])










if __name__ == "__main__":
    main()