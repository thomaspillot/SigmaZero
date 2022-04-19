# SigmaZero
<A level coursework project - Chess engine with integrated neural network>

The engine includes the functionality for both PvP and PvE game modes - with the PvE mode using a combination of minmax and the neural network to evaluate and automate moves for the black pieces. 
  
I did encounter a pretty major issue with the trained (and untrained) neural network - in which different board positions are evaluated to have the same 'value'. I wasn't able to pin point why exactly this happens - but I believe it is a mixture of an extremely sparse input (boards are encoded into bitboards before being passed), my potentially erroneous usage of GradientTape() (this was my first time using it), and just using hardly any training data (after parseGames.py filters and encodes the .pgn games into the hdf5 datsets, only around 50000 games were left). 

But anyway, I have included the trained model parameters in the file 'model1.txt' (I encountered some issues trying to save it as a tensorflow save file...), so once you have cloned the repo - just run the main.py file, which will automatically load the file's contents into the model (you may have to update some of the file paths first though). 

Since the PvE game mode of the engine utilises the neural network to make moves for the AI, you will also need to ensure that you have TensorFlow installed (I would recommend along with the CUDA/CUDnn dependencies met - so that computations can be carried out on your GPU). 
