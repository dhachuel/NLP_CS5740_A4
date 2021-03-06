## a4 starter repo

Logistics:

* You will edit the model.py and train.py functions to read in the .json files,
  train a sequence-to-sequence based model to predict a sequence of actions, and
  evaluate the actions by executing them against world states.
* An AlchemyWorldState is a class representing a possible alchemy world state.
  It can be initialized using the strings provided in the JSON files, e.g., 
  "1:_ 2:_ 3:_ 4:_ 5:_ 6:_ 7:_". It can also be manipulated by adding colors to
  a beaker or removing items from a beaker. 
* An AlchemyFSA is an interface from the action sequences to the world states.
  Given an AlchemyWorldState and an action in the form "push/pop # color/None",
  it can be used to either update the state (feed_complete_action),
  or just return the state that occurs when executing that action
  (peek_complete_action). The second case is useful if there is a chance the action
  will be invalid, e.g., popping from an empty beaker. 
* In general, the execute function in train.py shows how to take a string world
  state, create an AlchemyWorldState object, create an AlchemyFSA from that object,
  and then execute a sequence of actions.
* A shortcut to executing actions is to call execute_seq on an AlchemyWorldState.
  This will execute all actions in a sequence without having to create an AlchemyFSA.
  It will skip over invalid actions. 
* You should not need to edit any files except for model.py and train.py. You of course
  are welcome to split your code into multiple scripts.

Scripts you will need to edit:

* model.py: definition of your neural network model.
* train.py: functions for training the network and making predictions.

Utility scripts contained in this directory:

* alchemy_fsa.py: defines an FSA used to execute action sequences on a world state.
* alchemy_world_state.py: class for world states in the Alchemy domain that operate as a sequence of stacks.
* evaluate.py: script for evaluating CSV predicted states with CSV labels.
* extract_labels.py: used for generating the state labels for the training and development data.
* fsa.py: abstract class for FSA and world states.

Data files in the ''data'' and ''results'' directory:

* train.json, dev.json: contain the labeled training and development data.
* test.json: contains the unlabeled test inputs.
* *_instruction_y.csv, *_interaction_y.csv: labels for the training and development
    data on interaction-level and instruction-level final states.
* results/test_interaction_y.csv: placeholder for your predictions on the test set (interaction-level predictions).
