# SigmaZero
<A level coursework project - Chess engine with integrated neural network>

The engine includes the functionality for both PvP and PvE game modes - with the PvE mode using a combination of minmax and the neural network to evaluate and automate moves for the black pieces. 
  
I have included the trained model parameters in the file 'model1.txt' (I encountered some issues trying to save it as a tensorflow save file...), so once you have cloned the repo - just run the main.py file, which will automatically load the file's contents into the model (you may have to update some of the file paths first though). 

Since the PvE game mode of the engine utilises the neural network to make moves for the AI, you will also need to ensure that you have TensorFlow installed (I would recommend along with the CUDA/CUDnn dependencies met - so that computations can be carried out on your GPU). 
  
Also, as the saved model and pgn/hdf5 training data files are so large (combined they are around 300MB) - you may also need to install git LFS (large file storage) to pull them (I have configured .gitattributes to enable this functionality for the formats of said files - txt/PGN/HDF5).
