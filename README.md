# glossing-project

Final results reported here are located in the "finals" directory.

This code is provided primarily for documentary purposes. To use it, you will need an environment with langchain 0.1.15 and an OpenAI API key.

We assume that the SIGMORPHON challenge glossing data is located at 2023glossingST-main (clone from https://github.com/sigmorphon/2023glossingST --- note that the spelling is different for some reason).

To produce all glossing results for a language, run e.g.:

    python gloss_all.py --language Tsez --split dev --use_lm

To analyze the confusion sets and generate instructions for disambiguating a syncretic case, use:

    python most_confused.py --gold 2023glossingST-main/data/Tsez/ddo-dev-track1-uncovered --pred final.txt --language Tsez --cutoff 5 --exclude_empty_tag --use_lm

Then regenerate the prompts and rerun using gloss_all.py

To run oracle or other decoding experiments on an existing set of LM outputs, use:

    python create_final_only.py Tsez

(For this file, you need to edit the code to configure the decoding heuristic and the split name.)
