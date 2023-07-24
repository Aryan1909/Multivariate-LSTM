# Multivariate-LSTM
I made this LSTM model to predict the drift in my DLI (Delay Line Interferometer) by taking visibilities as input and giving out bias-voltage outputs. It can easily be used to predict some moderately complex data with good training...


Using this is very simple, just make different columns for all the inputs in the lstm in a .csv file and put the input column numbers in cols = list(df)[1:n+1] 
(suppose you want the lstm to take column 1 to n so you should put as cols = list(df)[1:n+1]).

Choose the n_past value to how much the LSTM should look back to make new predictions, you can add multiple lstms on top of each other. Here I used 2 LSTMS but
commented one out for simplicity, just make sure to change return_sequences after adding new lstms. (only the last lstm in the model should have return_sequences = True)
lastly, v_outs is the number of outputs you wanna predict. I also used a different code called "predictor" in this repo where I can put an input in the trained
lstm model to get an output.


Hope it helps. And you don't have to spend a week learning all this :))
