# DeepBS
TFT Board Strength Calculator

# TLDR 
Casual project for developing a neural network (Transformer + FC MLP) which takes as input two TFT boards and outputs a probability of the 1st board winning (0 - 1). 

Input features are derived from raw stats/text, so hopefully will allow this model to generalize across patches/sets. 

The reason for doing a pairwise win probability rather than single board strength score is b/c often outcomes are matchup dependent (rock-paper-scissors). 

# Method 
See summary slides [here](https://docs.google.com/presentation/d/1fYU9uPyvYCgMk84W6g4uxOgDL0LwqRZg7LAz2cafx7k/edit#slide=id.p). 

# Data
See raw jsons for game [here](https://raw.communitydragon.org/latest/cdragon/tft/en_us.json)
